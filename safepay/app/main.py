"""SafePay service: ISO 20022 screening API, analyst console API, demo.

Run on the LinuxONE VM:
  uvicorn safepay.app.main:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from safepay.core import evidence_pack, granite, iso20022, relmodel
from safepay.core.privacy import Privacy
from safepay.core.screener import Screener
from safepay.core.state import FEATURES
from safepay.core.store import Store
from safepay.app import security
from safepay.app.security import analyst, require_bank

ROOT = Path(__file__).resolve().parents[2]
ART, STATIC = ROOT / "artifacts", Path(__file__).parent / "static"
DB = Path(os.environ.get("SAFEPAY_DB", ROOT / "safepay.db"))

text_model = joblib.load(ART / "text.joblib")
rel_bundle = joblib.load(ART / "rel.joblib")
def _measured_fastest() -> str | None:
    """Use the engine the benchmark measured as fastest on this machine, not a guess."""
    f = ART / "benchmark.json"
    return json.loads(f.read_text()).get("fastest_tree_format") if f.exists() else None


TREE_FORMAT = relmodel.optimise(rel_bundle["model"], np.zeros((256, len(rel_bundle.get("features", FEATURES))), dtype=np.float32),
                                os.environ.get("SAFEPAY_TREE_FORMAT") or _measured_fastest())
rel_bundle["version"] = f"text+rel/{rel_bundle['kind']}/{time.strftime('%Y%m%d', time.gmtime((ART / 'rel.joblib').stat().st_mtime))}"
store = Store(DB)
privacy = Privacy()
SIGN_KEY = evidence_pack.signing_key()
app = FastAPI(title="SafePay", version="1.0", docs_url=None if security.settings.production else "/docs",
              redoc_url=None, openapi_url=None if security.settings.production else "/openapi.json")
app.middleware("http")(security.middleware)
GRANITE_QUEUE_MAX = 4      # background briefs waiting at most; more requests are refused, not queued

STATS = {"screened": 0, "hidden": 0, "cases_opened": 0, "nudges": 0}
LAT = deque(maxlen=5000)
GRANITE = {"jobs": 0, "last_seconds": None, "tokens_per_second": None}


# -- Granite background jobs -----------------------------------------------------
def case_facts(case_id: int) -> tuple[dict, list[str]]:
    c = store.case(case_id)
    st = screener.engine.states.get((c["sender"], c["recipient"]))
    ev = store.evidence(case_id)
    msgs = [privacy.open(e["sealed_ref"], f"case:{case_id}") for e in ev]
    if st is None:
        return {"payments_from_sender": len(ev)}, msgs
    return granite.facts_from_state(st, c["score"], json.loads(c["reasons"] or "[]")), msgs


def granite_job(case_id: int) -> None:
    if not granite.available():
        return
    facts, msgs = case_facts(case_id)
    t0 = time.time()
    text, timings = granite.summary(facts, msgs)
    opinion, _ = granite.second_opinion(facts, msgs)
    took = time.time() - t0
    store.set_case(case_id, summary=text, summary_by=f"IBM Granite 3.3 2B · llama.cpp on {platform.machine()} · {took:.0f}s",
                   second_opinion=json.dumps(opinion))
    GRANITE.update(jobs=GRANITE["jobs"] + 1, last_seconds=round(took, 1),
                   tokens_per_second=round(timings.get("predicted_per_second", 0) or 0, 2))
    store.audit("granite", "summarised", case_id, f"{took:.1f}s")


worker = granite.Worker(granite_job)
AUTO_GRANITE: set[str] = set()      # recipient tokens whose new cases get Granite automatically


def on_new_case(case_id: int) -> None:
    STATS["cases_opened"] += 1
    c = store.case(case_id)
    if c and c["recipient"] in AUTO_GRANITE and worker.q.qsize() < GRANITE_QUEUE_MAX:
        worker.submit((case_id,))


def on_escalate(case_id: int) -> None:
    """New severe evidence (a threat): the earlier brief is out of date, write a new one."""
    c = store.case(case_id)
    if c and c["recipient"] in AUTO_GRANITE and worker.q.qsize() < GRANITE_QUEUE_MAX:
        worker.submit((case_id,))


screener = Screener(text_model, rel_bundle, store, privacy, on_new_case=on_new_case, on_escalate=on_escalate)


def record(result: dict) -> dict:
    STATS["screened"] += 1
    STATS["hidden"] += result["decision"]["message"] == "HIDE"
    LAT.append(result["latency_ms"]["total"])
    return result


# -- bank-facing API -----------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model_version": screener.model_version, "tree_format": TREE_FORMAT}


@app.post("/v1/screen", dependencies=[Depends(require_bank)])
async def screen(request: Request) -> dict:
    """Body: an ISO 20022 pacs.008 or pain.001 document. Returns one decision per transaction."""
    body = await request.body()
    try:
        payments = iso20022.parse(body)
    except iso20022.UnsafeMessage as e:
        raise HTTPException(413 if "larger" in str(e) else 400, str(e))
    except Exception:
        raise HTTPException(400, "not a supported ISO 20022 message (pacs.008 or pain.001)")
    return {"results": [record(screener.screen(p)) for p in payments]}


class PreSubmit(BaseModel):
    sender_account: str
    recipient_account: str
    amount: float = Field(..., gt=0)
    reference: str = Field("", max_length=140)


@app.post("/v1/presubmit", dependencies=[Depends(require_bank)])
def presubmit(body: PreSubmit) -> dict:
    """Sender's bank, before the customer presses send. Never reveals that a case exists."""
    p = iso20022.new_payment("", body.sender_account, "", body.recipient_account, body.amount, body.reference)
    r = screener.screen(p, presubmit=True)
    STATS["nudges"] += r["decision"]["nudge_sender"]
    return {"nudge": r["decision"]["nudge_sender"],
            "message": "Payment messages must not be used to threaten or harass. Please edit your message."
            if r["decision"]["nudge_sender"] else ""}


class Hide(BaseModel):
    recipient_account: str
    sender_account: str


@app.post("/v1/recipients/hide", dependencies=[Depends(require_bank)])
def hide(body: Hide) -> dict:
    screener.hide_sender(body.recipient_account, body.sender_account)
    return {"ok": True}


# -- recipient safety: preferences and consent ---------------------------------------
class SafetyPrefs(BaseModel):
    contact_channel: str = Field("in_app", pattern="^(in_app|phone|email|none)$")
    safe_hours_from: str = Field("09:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    safe_hours_to: str = Field("17:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shared_device: bool = False          # "someone else may see my phone"
    note: str = Field("", max_length=200)


def save_prefs(recipient_token: str, p: SafetyPrefs) -> None:
    # Encrypted: a note like "my partner checks my phone" is itself sensitive.
    store.set_prefs(recipient_token, {"sealed": privacy.seal(p.model_dump_json(), f"prefs:{recipient_token}")})
    store.audit("recipient", "set_safety_preferences", None, recipient_token)


def load_prefs(recipient_token: str) -> dict | None:
    row = store.prefs(recipient_token)
    if not row:
        return None
    return {**json.loads(privacy.open(row["sealed"], f"prefs:{recipient_token}")), "updated": row["updated"]}


class PrefsIn(SafetyPrefs):
    recipient_account: str


@app.post("/v1/recipients/preferences", dependencies=[Depends(require_bank)])
def set_preferences(body: PrefsIn) -> dict:
    save_prefs(privacy.token(body.recipient_account), SafetyPrefs(**body.model_dump(exclude={"recipient_account"})))
    return {"ok": True}


class ConsentIn(BaseModel):
    recipient_account: str
    sender_account: str
    purpose: str = Field("police", pattern="^(police|protection_order|legal_advice)$")
    grant: bool = True


def set_consent(recipient_token: str, sender_token: str, purpose: str, grant: bool) -> dict:
    if grant:
        cid = store.grant_consent(recipient_token, sender_token, purpose)
        store.audit("recipient", f"consent_granted:{purpose}", None, f"{recipient_token} for {sender_token}")
        return {"ok": True, "consent_id": cid}
    store.withdraw_consent(recipient_token, sender_token)
    store.audit("recipient", "consent_withdrawn", None, f"{recipient_token} for {sender_token}")
    return {"ok": True}


@app.post("/v1/recipients/consent", dependencies=[Depends(require_bank)])
def consent(body: ConsentIn) -> dict:
    return set_consent(privacy.token(body.recipient_account), privacy.token(body.sender_account), body.purpose, body.grant)


# -- analyst API -------------------------------------------------------------------
def _case_view(c: dict) -> dict:
    c = dict(c)
    c["reasons"] = json.loads(c["reasons"] or "[]")
    c["second_opinion"] = json.loads(c["second_opinion"]) if c.get("second_opinion") else None
    c["granite_position"] = worker.position((c["id"],))
    return c


@app.get("/v1/cases", dependencies=[Depends(analyst)])
def cases(status: str | None = None) -> list[dict]:
    return [_case_view(c) for c in store.cases(status)]


@app.get("/v1/cases/{case_id}")
def case(case_id: int, x_analyst: str = Depends(analyst)) -> dict:
    c = store.case(case_id)
    if not c:
        raise HTTPException(404)
    store.audit(x_analyst, "viewed_case", case_id)
    ev = [{"ts": e["ts"], "amount": e["amount"], "harm": round(e["harm"], 3), "threat": round(e["threat"], 3),
           "hidden": bool(e["hidden"]), "reference": privacy.open(e["sealed_ref"], f"case:{case_id}")}
          for e in store.evidence(case_id)]
    st = screener.engine.states.get((c["sender"], c["recipient"]))
    facts, _ = case_facts(case_id)
    cons = store.consent(c["recipient"], c["sender"])
    safety = {"preferences": load_prefs(c["recipient"]),
              "consent": {"purpose": evidence_pack.PURPOSES[cons["purpose"]], "granted": cons["granted"]} if cons else None,
              "packs": store.packs(case_id), "evidence_log": store.verify_chain(case_id)}
    return {"case": _case_view(c), "evidence": ev, "facts": facts, "safety": safety,
            "hidden_at": st.hidden_at if st else 0, "first_ts": st.first_ts if st else None}


class Label(BaseModel):
    label: str = Field(..., pattern="^(abuse|not_abuse)$")


@app.post("/v1/cases/{case_id}/label")
def label(case_id: int, body: Label, x_analyst: str = Depends(analyst)) -> dict:
    c = store.case(case_id)
    if not c:
        raise HTTPException(404)
    st = screener.engine.states.get((c["sender"], c["recipient"]))
    store.label(case_id, x_analyst, body.label,
                screener.engine.features(c["sender"], c["recipient"], st, st.last_ts) if st else [])
    store.audit(x_analyst, f"labelled_{body.label}", case_id)
    return {"ok": True, "labels_for_retraining": store.q("SELECT COUNT(*) n FROM labels")[0]["n"]}


@app.post("/v1/cases/{case_id}/granite", dependencies=[Depends(analyst)])
def ask_granite(case_id: int) -> dict:
    if not store.case(case_id):
        raise HTTPException(404)
    if worker.position((case_id,)) is None:
        if worker.q.qsize() >= GRANITE_QUEUE_MAX:
            raise HTTPException(429, "Granite is busy with other cases; try again in a few minutes.")
        worker.submit((case_id,))
    # position 0 = being written now, n = n briefs ahead of it
    return {"queued": True, "position": worker.position((case_id,)), "queue_length": worker.q.qsize()}


@app.post("/v1/cases/{case_id}/evidence-pack")
def make_pack(case_id: int, x_analyst: str = Depends(analyst)) -> dict:
    c = store.case(case_id)
    if not c:
        raise HTTPException(404)
    cons = store.consent(c["recipient"], c["sender"])
    if not cons:
        raise HTTPException(403, "The recipient has not consented to a record of this sender's payments.")
    chain = store.verify_chain(case_id)
    if not chain["intact"]:
        store.audit(x_analyst, "pack_refused_integrity", case_id, f"bad entry {chain['first_bad_entry']}")
        raise HTTPException(409, f"Evidence log fails its integrity check at entry {chain['first_bad_entry']}.")
    ev = [{**e, "reference": privacy.open(e["sealed_ref"], f"case:{case_id}")} for e in store.evidence(case_id)]
    facts, _ = case_facts(case_id)
    body = evidence_pack.build(c, ev, facts, cons, chain, x_analyst, screener.model_version,
                               c.get("summary"), c.get("summary_by"))
    signed = evidence_pack.sign(body, SIGN_KEY)
    pid = store.add_pack(case_id, x_analyst, signed["sha256"],
                         privacy.seal(json.dumps(signed), f"pack:{signed['sha256']}"), cons["id"])
    store.audit(x_analyst, "evidence_pack_created", case_id, signed["sha256"])
    return {"pack_id": pid, "sha256": signed["sha256"], "url": f"/pack/{pid}"}


@app.get("/v1/packs/{pack_id}")
def get_pack(pack_id: int, x_analyst: str = Depends(analyst)) -> dict:
    p = store.pack(pack_id)
    if not p:
        raise HTTPException(404)
    store.audit(x_analyst, "evidence_pack_viewed", p["case_id"], p["sha256"])
    return json.loads(privacy.open(p["sealed_body"], f"pack:{p['sha256']}"))


@app.get("/v1/evidence/public-key")
def public_key() -> dict:
    return {"algorithm": "Ed25519", "public_key": evidence_pack.public_key_hex(SIGN_KEY)}


@app.post("/v1/packs/verify")
async def verify_pack(request: Request) -> dict:
    """Anyone holding a pack (police, a court) can check it. Nothing is stored."""
    try:
        signed = await request.json()
    except Exception:
        raise HTTPException(400, "body must be the pack's JSON")
    out = evidence_pack.verify(signed, evidence_pack.public_key_hex(SIGN_KEY))
    known = store.pack_by_hash(signed.get("sha256", "")) if isinstance(signed, dict) else None
    out["issued_by_this_service"] = known is not None
    if known:
        out["evidence_log_still_intact"] = store.verify_chain(known["case_id"])["intact"]
    return out


@app.get("/v1/audit", dependencies=[Depends(analyst)])
def audit() -> list[dict]:
    return store.audit_log()


# -- system / metrics ----------------------------------------------------------
IBM_Z = {"9175": "IBM z17", "9176": "IBM z17", "3931": "IBM z16", "3932": "IBM z16", "8561": "IBM z15", "8562": "IBM z15"}


@app.get("/api/system")
def system() -> dict:
    cpu = "unknown"
    try:
        info = Path("/proc/cpuinfo").read_text()
        m = re.search(r"machine\s*=\s*(\d{4})", info)
        cpu = f"{IBM_Z.get(m.group(1), 'IBM Z')} (machine type {m.group(1)})" if m else cpu
        nnpa = " nnpa" in info
    except OSError:
        nnpa = False
    lat = sorted(LAT)
    pct = lambda q: round(lat[min(int(q * len(lat)), len(lat) - 1)], 3) if lat else None
    return {"machine": platform.machine(), "byteorder": sys.byteorder, "cpu": cpu, "cores": os.cpu_count(),
            "ai_accelerator_present": nnpa, "tree_format": TREE_FORMAT, "python": platform.python_version(),
            "granite": {"available": granite.available(), **GRANITE, "queue": worker.q.qsize(),
                        "error": worker.last_error},
            "live": {**STATS, "latency_ms_p50": pct(0.5), "latency_ms_p99": pct(0.99)},
            "model_version": screener.model_version}


@app.get("/api/model-metrics")
def model_metrics() -> dict:
    out = {}
    for name in ("metrics_text", "metrics_text_unmasked", "metrics_rel", "benchmark", "cba_validation",
                 "granite_eval_v1_seed6", "granite_eval_v2_seed6", "granite_eval_v2_seed7",
                 "eval_prevalence", "eval_test_protect", "eval_before_switch_val", "eval_before_switch_test",
                 "metrics_rel_paironly"):
        f = ART / f"{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else None
    return out


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_bank)])
def prometheus() -> str:
    s = system()["live"]
    return "\n".join([f"safepay_payments_screened_total {STATS['screened']}",
                      f"safepay_messages_hidden_total {STATS['hidden']}",
                      f"safepay_cases_opened_total {STATS['cases_opened']}",
                      f"safepay_sender_nudges_total {STATS['nudges']}",
                      f"safepay_latency_ms{{quantile=\"0.5\"}} {s['latency_ms_p50'] or 0}",
                      f"safepay_latency_ms{{quantile=\"0.99\"}} {s['latency_ms_p99'] or 0}", ""])


# -- demo: two banking apps and scripted scenarios --------------------------------
PEOPLE = {
    "jordan": ("Jordan Reid", "062000-10442871"),
    "sam": ("Sam Patel", "063010-55519230"),
    "alex": ("Alex Chen", "062500-77134002"),
    "kim": ("Kim Taylor", "064000-23981145"),
    "riley": ("Riley Morgan", "062100-38820417"),
    "chris": ("Chris Walker", "063200-90155362"),
    "casey": ("Casey Brooks", "062400-58823190"),
    "drew": ("Drew Kelly", "062300-41177208"),     # the account Casey switches to
}
INBOX: dict[str, list[dict]] = {k: [] for k in PEOPLE}
DEMO_CLOCK = {"t": None}
for name, (_, acct) in PEOPLE.items():
    if name in ("sam", "kim"):
        AUTO_GRANITE.add(privacy.token(acct))


def _demo_time(step_minutes: float | None) -> datetime:
    """Live sends use real time; scripted scenarios advance a simulated clock."""
    if step_minutes is None:
        return datetime.now(timezone.utc)
    base = DEMO_CLOCK["t"] or datetime.now(timezone.utc).replace(hour=1, minute=10, second=0, microsecond=0)
    DEMO_CLOCK["t"] = base + timedelta(minutes=step_minutes)
    return DEMO_CLOCK["t"]


def demo_send(frm: str, to: str, amount: float, reference: str, step_minutes: float | None = None) -> dict:
    (sn, sa), (rn, ra) = PEOPLE[frm], PEOPLE[to]
    p = iso20022.new_payment(sn, sa, rn, ra, amount, reference, when=_demo_time(step_minutes))
    xml = iso20022.build_pacs008(p)
    res = record(screener.screen(iso20022.parse(xml)[0]))
    INBOX[to].insert(0, {"from": sn, "from_key": frm, "amount": amount, "ts": p.created.isoformat(),
                         "reference": reference, "hidden": res["decision"]["message"] == "HIDE",
                         "reasons": res["decision"]["reasons"]})
    # Split messages: once the joined text is harmful, earlier fragments are hidden in the app too.
    if res["scores"]["joined_recent"] and res["decision"]["message"] == "HIDE":
        n = len(res["scores"]["joined_recent"].split()) - len(reference.split())
        words = 0
        for item in INBOX[to][1:]:
            if item["from_key"] != frm or words >= n:
                break
            item["hidden"], words = True, words + len(item["reference"].split())
    res["iso20022_in"] = xml
    return res


class Send(BaseModel):
    sender: str
    recipient: str
    amount: float = Field(..., gt=0, le=100000)
    reference: str = Field("", max_length=140)


@app.post("/demo/send")
def demo_send_api(body: Send) -> dict:
    if body.sender not in PEOPLE or body.recipient not in PEOPLE or body.sender == body.recipient:
        raise HTTPException(400, "unknown people")
    pre = presubmit(PreSubmit(sender_account=PEOPLE[body.sender][1], recipient_account=PEOPLE[body.recipient][1],
                              amount=body.amount, reference=body.reference))
    res = demo_send(body.sender, body.recipient, body.amount, body.reference)
    res["presubmit"] = pre
    return res


@app.get("/demo/inbox/{person}")
def inbox(person: str) -> list[dict]:
    return INBOX.get(person, [])[:40]


@app.post("/demo/hide/{recipient}/{sender}")
def demo_hide(recipient: str, sender: str) -> dict:
    screener.hide_sender(PEOPLE[recipient][1], PEOPLE[sender][1])
    return {"ok": True}


def _person(key: str) -> tuple[str, str]:
    if key not in PEOPLE:
        raise HTTPException(404, "unknown person")
    return PEOPLE[key]


@app.get("/demo/safety/{person}")
def demo_safety(person: str) -> dict:
    """What the recipient's 'Message settings' screen shows."""
    _, acct = _person(person)
    r_tok = privacy.token(acct)
    hidden: dict[str, int] = {}
    for item in INBOX[person]:
        if item["hidden"]:
            hidden[item["from_key"]] = hidden.get(item["from_key"], 0) + 1
    senders = []
    for key, n in hidden.items():
        s_tok = privacy.token(PEOPLE[key][1])
        cons = store.consent(r_tok, s_tok)
        senders.append({"key": key, "name": PEOPLE[key][0], "hidden_messages": n,
                        "consent": cons["purpose"] if cons else None})
    return {"preferences": load_prefs(r_tok) or SafetyPrefs().model_dump(), "senders": senders,
            "purposes": evidence_pack.PURPOSES}


@app.post("/demo/safety/{person}/preferences")
def demo_prefs(person: str, body: SafetyPrefs) -> dict:
    save_prefs(privacy.token(_person(person)[1]), body)
    return {"ok": True}


class DemoConsent(BaseModel):
    sender: str
    purpose: str = Field("police", pattern="^(police|protection_order|legal_advice)$")
    grant: bool = True


@app.post("/demo/safety/{person}/consent")
def demo_consent(person: str, body: DemoConsent) -> dict:
    return set_consent(privacy.token(_person(person)[1]), privacy.token(_person(body.sender)[1]), body.purpose, body.grant)


# Each scenario has its own sender, so cases stay separate in the review queue.
SCRIPTS = {
    # The case reported by the press: one threat spread over nine transfers.
    "split_threat": ("jordan", "sam", [(0.01, w) for w in "I am going to find you and kill you".split()]),
    "coercive": ("riley", "sam", [(0.01, "unblock me"), (0.01, "answer your phone"), (0.01, "saw you at the gym today"),
                                  (0.05, "nice top"), (0.01, "why are you ignoring me"), (0.01, "I'm outside"),
                                  (0.01, "you can't hide from me"), (0.01, "open the door")]),
    "child_support": ("chris", "kim", [(350, "child support"), (350, "for the kids"),
                                       (350, "you will never see them again you worthless b1tch"),
                                       (350, "child support"), (350, "unless you come back this is the last")]),
    "friends": ("alex", "kim", [(24.5, "pizza 🍕"), (12, "you owe me loser haha"), (60, "concert tickets"),
                                (8, "killing it at work congrats"), (30, "dinner, you pig"), (15, "uber")]),
    # Hidden by Sam, Casey carries on from a new account. Steps may name their own sender, or hide one.
    "account_switch": ("casey", "sam", [(0.01, "pick up"), (0.01, "where are you"), (0.01, "answer me"),
                                        ("hide", "casey"),
                                        ("drew", 0.01, "new account, you can't block me"), ("drew", 0.01, "answer me"),
                                        ("drew", 0.01, "I'm outside")]),
}


@app.post("/demo/script/{name}")
def script(name: str) -> dict:
    if name not in SCRIPTS:
        raise HTTPException(404)
    frm, to, steps = SCRIPTS[name]
    gap = {"split_threat": 0.5, "coercive": 37.0, "child_support": 14 * 24 * 60.0, "friends": 3 * 24 * 60.0,
           "account_switch": 45.0}[name]
    out = []
    for step in steps:
        if step[0] == "hide":
            screener.engine.hide(privacy.token(PEOPLE[to][1]), privacy.token(PEOPLE[step[1]][1]),
                                 _demo_time(gap).timestamp())
            store.hide(privacy.token(PEOPLE[to][1]), privacy.token(PEOPLE[step[1]][1]), DEMO_CLOCK["t"].timestamp())
            continue
        sender, amt, ref = step if len(step) == 3 else (frm, *step)
        r = demo_send(sender, to, amt, ref, step_minutes=gap)
        out.append({"sender": PEOPLE[sender][0], "reference": ref, "amount": amt, "decision": r["decision"],
                    "scores": r["scores"], "case_id": r["case_id"], "latency_ms": r["latency_ms"], "_full": r})
    last = out[-1].pop("_full")
    for o in out:
        o.pop("_full", None)
    return {"steps": out, "last": last}


@app.post("/demo/reset")
def reset() -> dict:
    store.reset()
    screener.engine.states.clear()
    for k in INBOX:
        INBOX[k].clear()
    DEMO_CLOCK["t"] = None
    for k in STATS:
        STATS[k] = 0
    LAT.clear()
    return {"ok": True}


BACKGROUND = {"running": False, "done": 0, "total": 0}


@app.post("/demo/background")
def background(n_innocent: int = 1500, n_abusive: int = 120) -> dict:
    """Replay a fresh synthetic world through the real screening path to fill the review queue."""
    if BACKGROUND["running"]:
        return BACKGROUND

    def run():
        from safepay.sim import world as W
        w = W.build("test", n_innocent, n_abusive, seed=int(time.time()) % 10_000)
        BACKGROUND.update(running=True, done=0, total=len(w.txs))
        hides = list(w.hides)
        hi = 0
        for tx in w.txs:
            while hi < len(hides) and hides[hi][0] <= tx.ts:
                _, r, s = hides[hi]
                screener.hide_sender(r, s)
                hi += 1
            p = iso20022.new_payment(w.customers[tx.sender], tx.sender, w.customers[tx.recipient], tx.recipient,
                                     tx.amount, tx.reference, when=tx.ts)
            record(screener.screen(p))
            BACKGROUND["done"] += 1
        BACKGROUND["running"] = False

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.2)
    return BACKGROUND


@app.get("/demo/background")
def background_status() -> dict:
    return BACKGROUND


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/analyst")
def analyst_page() -> FileResponse:
    return FileResponse(STATIC / "analyst.html")


@app.get("/system")
def system_page() -> FileResponse:
    return FileResponse(STATIC / "system.html")


@app.get("/pack/{pack_id}")
def pack_page(pack_id: int) -> FileResponse:
    return FileResponse(STATIC / "pack.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
