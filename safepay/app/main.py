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
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from safepay.core import granite, iso20022, relmodel
from safepay.core.privacy import Privacy
from safepay.core.screener import Screener
from safepay.core.state import FEATURES
from safepay.core.store import Store

ROOT = Path(__file__).resolve().parents[2]
ART, STATIC = ROOT / "artifacts", Path(__file__).parent / "static"
DB = Path(os.environ.get("SAFEPAY_DB", ROOT / "safepay.db"))

text_model = joblib.load(ART / "text.joblib")
rel_bundle = joblib.load(ART / "rel.joblib")
def _measured_fastest() -> str | None:
    """Use the engine the benchmark measured as fastest on this machine, not a guess."""
    f = ART / "benchmark.json"
    return json.loads(f.read_text()).get("fastest_tree_format") if f.exists() else None


TREE_FORMAT = relmodel.optimise(rel_bundle["model"], np.zeros((256, len(FEATURES)), dtype=np.float32),
                                os.environ.get("SAFEPAY_TREE_FORMAT") or _measured_fastest())
rel_bundle["version"] = f"text+rel/{rel_bundle['kind']}/{time.strftime('%Y%m%d', time.gmtime((ART / 'rel.joblib').stat().st_mtime))}"
store = Store(DB)
privacy = Privacy()
app = FastAPI(title="SafePay", version="1.0")

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
    opinion, t2 = granite.second_opinion(facts, msgs)
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
    if c and c["recipient"] in AUTO_GRANITE:
        worker.submit((case_id,))


def on_escalate(case_id: int) -> None:
    """New severe evidence (a threat): the earlier brief is out of date, write a new one."""
    c = store.case(case_id)
    if c and c["recipient"] in AUTO_GRANITE:
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


@app.post("/v1/screen")
async def screen(request: Request) -> dict:
    """Body: an ISO 20022 pacs.008 or pain.001 document. Returns one decision per transaction."""
    try:
        payments = iso20022.parse(await request.body())
    except Exception as e:
        raise HTTPException(400, f"not a supported ISO 20022 message: {e}")
    return {"results": [record(screener.screen(p)) for p in payments]}


class PreSubmit(BaseModel):
    sender_account: str
    recipient_account: str
    amount: float = Field(..., gt=0)
    reference: str = Field("", max_length=140)


@app.post("/v1/presubmit")
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


@app.post("/v1/recipients/hide")
def hide(body: Hide) -> dict:
    screener.hide_sender(body.recipient_account, body.sender_account)
    return {"ok": True}


# -- analyst API -------------------------------------------------------------------
def _case_view(c: dict) -> dict:
    c = dict(c)
    c["reasons"] = json.loads(c["reasons"] or "[]")
    c["second_opinion"] = json.loads(c["second_opinion"]) if c.get("second_opinion") else None
    return c


@app.get("/v1/cases")
def cases(status: str | None = None) -> list[dict]:
    return [_case_view(c) for c in store.cases(status)]


@app.get("/v1/cases/{case_id}")
def case(case_id: int, x_analyst: str = Header("analyst-demo")) -> dict:
    c = store.case(case_id)
    if not c:
        raise HTTPException(404)
    store.audit(x_analyst, "viewed_case", case_id)
    ev = [{"ts": e["ts"], "amount": e["amount"], "harm": round(e["harm"], 3), "threat": round(e["threat"], 3),
           "hidden": bool(e["hidden"]), "reference": privacy.open(e["sealed_ref"], f"case:{case_id}")}
          for e in store.evidence(case_id)]
    st = screener.engine.states.get((c["sender"], c["recipient"]))
    facts, _ = case_facts(case_id)
    return {"case": _case_view(c), "evidence": ev, "facts": facts,
            "hidden_at": st.hidden_at if st else 0, "first_ts": st.first_ts if st else None}


class Label(BaseModel):
    label: str = Field(..., pattern="^(abuse|not_abuse)$")


@app.post("/v1/cases/{case_id}/label")
def label(case_id: int, body: Label, x_analyst: str = Header("analyst-demo")) -> dict:
    c = store.case(case_id)
    if not c:
        raise HTTPException(404)
    st = screener.engine.states.get((c["sender"], c["recipient"]))
    store.label(case_id, x_analyst, body.label, st.features(st.last_ts) if st else [])
    store.audit(x_analyst, f"labelled_{body.label}", case_id)
    return {"ok": True, "labels_for_retraining": store.q("SELECT COUNT(*) n FROM labels")[0]["n"]}


@app.post("/v1/cases/{case_id}/granite")
def ask_granite(case_id: int) -> dict:
    if not store.case(case_id):
        raise HTTPException(404)
    worker.submit((case_id,))
    return {"queued": True, "queue_length": worker.q.qsize()}


@app.get("/v1/audit")
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
                 "granite_eval_v1_seed6", "granite_eval_v2_seed6", "granite_eval_v2_seed7"):
        f = ART / f"{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else None
    return out


@app.get("/metrics", response_class=PlainTextResponse)
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
    "friends": ("alex", "sam", [(24.5, "pizza 🍕"), (12, "you owe me loser haha"), (60, "concert tickets"),
                                (8, "killing it at work congrats"), (30, "dinner, you pig"), (15, "uber")]),
}


@app.post("/demo/script/{name}")
def script(name: str) -> dict:
    if name not in SCRIPTS:
        raise HTTPException(404)
    frm, to, steps = SCRIPTS[name]
    gap = {"split_threat": 0.5, "coercive": 37.0, "child_support": 14 * 24 * 60.0, "friends": 3 * 24 * 60.0}[name]
    results = [demo_send(frm, to, amt, ref, step_minutes=gap) for amt, ref in steps]
    return {"steps": [{"reference": ref, "amount": amt, "decision": r["decision"], "scores": r["scores"],
                       "case_id": r["case_id"], "latency_ms": r["latency_ms"]}
                      for (amt, ref), r in zip(steps, results)], "last": results[-1]}


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


app.mount("/static", StaticFiles(directory=STATIC), name="static")
