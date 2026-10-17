"""SafePay tests. Unit tests run anywhere; end-to-end tests need trained
artifacts (on the VM) and are skipped without them.

Run: python3 -m pytest -q tests_safepay
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from safepay.core import iso20022
from safepay.core.engine import Engine, informative
from safepay.core.normalize import canonicalize
from safepay.core.policy import MIN_EVIDENCE, Thresholds, decide
from safepay.core.privacy import Privacy
from safepay.core.state import SPLIT_WINDOW, PairState
from safepay.core.textmodel import mask_identity

ART = Path(__file__).resolve().parents[1] / "artifacts"
needs_models = pytest.mark.skipif(not (ART / "rel.joblib").exists(), reason="models not trained")
T0 = datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc)


class KeywordModel:
    """Stand-in message model: 'kill' is a threat, 'idiot' is abusive."""
    thresholds = {"harm": 0.5, "threat": 0.5}

    def proba(self, texts):
        t = [canonicalize(x) for x in texts]
        return {"harm": np.array([0.9 if "idiot" in x or "kill" in x else 0.05 for x in t]),
                "threat": np.array([0.95 if "kill" in x else 0.02 for x in t])}


class ConstRel:
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return np.array([[1 - self.p, self.p]] * len(X))


# -- ISO 20022 -------------------------------------------------------------------
def test_pacs008_roundtrip_and_withheld_reference():
    p = iso20022.new_payment("Jordan Reid", "062000-1", "Sam Patel", "063010-2", 0.01, "see you soon", when=T0)
    back = iso20022.parse(iso20022.build_pacs008(p))[0]
    assert (back.amount, back.reference, back.debtor_account, back.creditor_account) == (0.01, "see you soon", "062000-1", "063010-2")
    hidden = iso20022.parse(iso20022.with_reference(p, iso20022.WITHHELD))[0]
    assert hidden.reference == iso20022.WITHHELD and hidden.amount == 0.01 and hidden.end_to_end_id == p.end_to_end_id


def test_pain001_parse():
    ns = iso20022.NS["pain.001"]
    xml = f"""<Document xmlns="{ns}"><CstmrCdtTrfInitn><GrpHdr><MsgId>M1</MsgId><CreDtTm>2026-06-01T10:00:00+00:00</CreDtTm></GrpHdr>
      <PmtInf><Dbtr><Nm>A</Nm></Dbtr><DbtrAcct><Id><Othr><Id>111</Id></Othr></Id></DbtrAcct>
      <CdtTrfTxInf><PmtId><EndToEndId>E1</EndToEndId></PmtId><Amt><InstdAmt Ccy="AUD">5.00</InstdAmt></Amt>
      <Cdtr><Nm>B</Nm></Cdtr><CdtrAcct><Id><IBAN>GB00TEST</IBAN></Id></CdtrAcct><RmtInf><Ustrd>rent</Ustrd></RmtInf></CdtTrfTxInf>
      </PmtInf></CstmrCdtTrfInitn></Document>"""
    p = iso20022.parse(xml)[0]
    assert (p.kind, p.amount, p.debtor_account, p.creditor_account, p.reference) == ("pain.001", 5.0, "111", "GB00TEST", "rent")


def test_rejects_unknown_namespace():
    with pytest.raises(ValueError):
        iso20022.parse("<Document xmlns='urn:x'><A/></Document>")


# -- text handling ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expect", [
    ("I will k.i.l.l you", "kill"), ("ur d3ad", "dead"), ("уоu", "you"), ("see you 🔪", "knife"), ("diiiiie", "diie")])
def test_canonicalize_undoes_evasion(raw, expect):
    assert expect in canonicalize(raw)


def test_identity_terms_masked_once():
    assert mask_identity("proud african american woman") == "proud identityterm"
    assert mask_identity("rent for march") == "rent for march"


def test_tiny_references_are_not_judged_alone():
    assert not informative("I") and not informative("you") and informative("kill") and informative("🔪")


# -- privacy ------------------------------------------------------------------------
def test_tokens_stable_and_evidence_bound_to_case():
    pv = Privacy(os.urandom(32))
    assert pv.token("062000-1") == pv.token(" 062000-1 ") != pv.token("062000-2")
    blob = pv.seal("you can't hide", "case:1")
    assert pv.open(blob, "case:1") == "you can't hide"
    with pytest.raises(Exception):
        pv.open(blob, "case:2")


# -- relationship state ------------------------------------------------------------
def test_split_messages_join_within_window_only():
    st = PairState()
    st.update(T0.timestamp(), 1, 0.01, "I am going", 0, 0)
    assert st.split_text(T0.timestamp() + 60, "to find you") == "I am going to find you"
    assert st.split_text(T0.timestamp() + SPLIT_WINDOW + 1, "later") is None


def test_engine_counts_replies_and_contact_after_hide():
    eng = Engine(KeywordModel(), ConstRel(0.1))
    eng.observe(T0, "a", "b", 0.01, "hello there")
    eng.observe(T0 + timedelta(minutes=5), "b", "a", 0.01, "please stop")
    eng.hide("b", "a", (T0 + timedelta(minutes=10)).timestamp())
    obs = eng.observe(T0 + timedelta(minutes=20), "a", "b", 0.01, "answer me")
    assert obs.state.n_reply == 1 and obs.state.n_after_hide == 1


def test_new_account_after_a_hide_is_not_a_stranger():
    eng = Engine(KeywordModel(), ConstRel(0.1))
    eng.observe(T0, "abuser", "victim", 0.01, "answer me")
    eng.hide("victim", "abuser", (T0 + timedelta(hours=1)).timestamp())
    obs = eng.observe(T0 + timedelta(days=2), "new_account", "victim", 0.01, "you can't hide")
    hid, gap_days, new_senders = obs.features[-3:]
    assert hid == 1.0 and 1.9 < gap_days < 2.0 and new_senders == 1.0
    # A sender who was already paying before the hide is not counted as new.
    early = eng.observe(T0 + timedelta(days=3), "abuser", "victim", 0.01, "hello")
    assert early.features[-3:] == [0.0, 60.0, 0.0]          # the hide was of this sender itself


def test_pair_only_engine_matches_older_models():
    eng = Engine(KeywordModel(), ConstRel(0.1), recipient_context=False)
    from safepay.core.state import PAIR_FEATURES
    assert len(eng.observe(T0, "a", "b", 5.0, "rent").features) == len(PAIR_FEATURES)


def test_split_threat_detected_when_joined():
    eng = Engine(KeywordModel(), ConstRel(0.1))
    for i, w in enumerate(["I will", "find you", "and kill", "you"]):
        obs = eng.observe(T0 + timedelta(minutes=i), "a", "b", 0.01, w)
    assert obs.harm == 0.0                        # "you" alone is too short to judge
    assert obs.split_text == "I will find you and kill you" and obs.split_threat > 0.5


# -- policy -------------------------------------------------------------------------
TH = Thresholds(harm=0.5, threat=0.5, case=0.6)


def test_money_always_delivered_and_abusive_message_hidden():
    d = decide(TH, harm=0.95, threat=0.1, split_harm=0, split_threat=0, rel_score=0.9, under_review=False,
               recipient_hid_sender=False)
    assert d.deliver_funds and d.message == "HIDE" and d.case == "open"


def test_no_behaviour_only_case_before_minimum_evidence():
    early = decide(TH, 0.0, 0.0, 0, 0, rel_score=0.9, under_review=False, recipient_hid_sender=False, n_payments=1)
    later = decide(TH, 0.0, 0.0, 0, 0, rel_score=0.9, under_review=False, recipient_hid_sender=False,
                   n_payments=MIN_EVIDENCE)
    assert early.case == "none" and later.case == "open"


def test_threat_is_top_priority():
    d = decide(TH, 0.2, 0.9, 0, 0, rel_score=0.7, under_review=False, recipient_hid_sender=False, n_payments=1)
    assert d.case == "open" and d.priority == "P1"


def test_sender_never_nudged_once_under_review():
    first = decide(TH, 0.9, 0.0, 0, 0, 0.1, under_review=False, recipient_hid_sender=False, presubmit=True)
    watched = decide(TH, 0.9, 0.0, 0, 0, 0.1, under_review=True, recipient_hid_sender=False, presubmit=True)
    assert first.nudge_sender and not watched.nudge_sender


# -- evidence log and evidence packs ------------------------------------------------
from safepay.core import evidence_pack  # noqa: E402
from safepay.core.store import Store  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402


def _store_with_evidence(n=3) -> Store:
    s = Store(":memory:")
    for i in range(n):
        s.add_evidence(7, 1_780_000_000 + i * 60, 0.01, f"sealed-{i}", 0.1 * i, 0.0, True)
    return s


def test_evidence_chain_detects_edits_and_deletions():
    s = _store_with_evidence()
    assert s.verify_chain(7)["intact"] and s.verify_chain(7)["entries"] == 3
    s.x("UPDATE evidence SET sealed_ref='changed' WHERE id=2")
    assert s.verify_chain(7) == {"intact": False, "entries": 3, "first_bad_entry": 2, "head": None}
    s2 = _store_with_evidence()
    s2.x("DELETE FROM evidence WHERE id=2")
    assert s2.verify_chain(7)["first_bad_entry"] == 3


def test_consent_grant_and_withdraw():
    s = Store(":memory:")
    assert s.consent("r", "s") is None
    s.grant_consent("r", "s", "police")
    assert s.consent("r", "s")["purpose"] == "police"
    s.withdraw_consent("r", "s")
    assert s.consent("r", "s") is None


def _signed_pack(key):
    s = _store_with_evidence()
    case = {"id": 7, "sender": "acct_s", "recipient": "acct_r", "opened": 1_780_000_000, "priority": "P1",
            "status": "open", "score": 0.97, "reasons": '["message reads as a threat (0.99)"]'}
    ev = [{**e, "reference": f"message {i}"} for i, e in enumerate(s.evidence(7))]
    consent = {"id": 1, "purpose": "protection_order", "granted": 1_780_000_500}
    body = evidence_pack.build(case, ev, {"payments_from_sender": 3, "model_risk_score": 0.97}, consent,
                               s.verify_chain(7), "analyst-1", "test")
    return evidence_pack.sign(body, key)


def test_pack_is_signed_and_any_change_is_detected():
    key = Ed25519PrivateKey.generate()
    pub = evidence_pack.public_key_hex(key)
    signed = _signed_pack(key)
    assert evidence_pack.verify(signed, pub)["valid"]
    assert "model_risk_score" not in signed["pack"]["pattern"]           # model output kept separate from facts
    assert signed["pack"]["purpose"] == "Application for a protection order"
    signed["pack"]["timeline"][1]["message"] = "edited"
    v = evidence_pack.verify(signed, pub)
    assert not v["valid"] and not v["fingerprint_matches"] and not v["signature_valid"]


def test_pack_signed_by_another_key_is_rejected():
    signed = _signed_pack(Ed25519PrivateKey.generate())
    v = evidence_pack.verify(signed, evidence_pack.public_key_hex(Ed25519PrivateKey.generate()))
    assert not v["valid"] and not v["signed_by_this_bank"]
    assert not evidence_pack.verify({"garbage": True}, "00" * 32)["valid"]


# -- end to end (trained models, on the VM) -----------------------------------------
@pytest.fixture(scope="module")
def api():
    tmp = Path(tempfile.mkdtemp())
    os.environ.update(SAFEPAY_DB=str(tmp / "t.db"), SAFEPAY_KEY_FILE=str(tmp / "key"),
                      SAFEPAY_SIGN_KEY_FILE=str(tmp / "sign_key"), SAFEPAY_LLM_URL="http://127.0.0.1:9")
    from fastapi.testclient import TestClient
    from safepay.app.main import app
    return TestClient(app)


@needs_models
def test_api_scenarios_end_to_end(api):
    c = api
    c.post("/demo/reset")
    split = c.post("/demo/script/split_threat").json()["last"]
    assert split["decision"]["message"] == "HIDE" and split["decision"]["priority"] == "P1"
    friends = c.post("/demo/script/friends").json()["steps"]
    assert all(s["case_id"] is None for s in friends)
    support = c.post("/demo/script/child_support").json()["steps"]
    assert all(s["decision"]["deliver_funds"] for s in support) and any(s["case_id"] for s in support)
    # Hidden, the abuser switches account: the new account's very first message is withheld.
    drew = [s for s in c.post("/demo/script/account_switch").json()["steps"] if s["sender"] == "Drew Kelly"]
    assert drew[0]["decision"]["message"] == "HIDE"
    assert any("new sender soon after" in r for r in drew[0]["decision"]["reasons"])
    xml = iso20022.build_pacs008(iso20022.new_payment("A", "1", "B", "2", 20.0, "rent"))
    r = c.post("/v1/screen", content=xml, headers={"Content-Type": "application/xml"}).json()["results"][0]
    assert r["decision"]["message"] == "SHOW" and "<Ustrd>rent</Ustrd>" in r["forwarded_pacs008"]
    assert c.get("/healthz").json()["ok"]


@needs_models
def test_safety_journey_consent_gates_a_verifiable_pack(api):
    c = api
    c.post("/demo/reset")
    case_id = c.post("/demo/script/coercive").json()["last"]["case_id"]
    assert case_id
    # No consent: no pack.
    assert c.post(f"/v1/cases/{case_id}/evidence-pack").status_code == 403
    # Recipient sets safe-contact preferences and consents for this one sender.
    note = "partner checks my phone"
    assert c.post("/demo/safety/sam/preferences", json={"contact_channel": "in_app", "safe_hours_from": "10:00",
                                                        "safe_hours_to": "14:00", "shared_device": True,
                                                        "note": note}).json()["ok"]
    safety = c.get("/demo/safety/sam").json()
    assert [x["key"] for x in safety["senders"]] == ["riley"] and safety["senders"][0]["consent"] is None
    c.post("/demo/safety/sam/consent", json={"sender": "riley", "purpose": "police"})
    detail = c.get(f"/v1/cases/{case_id}").json()["safety"]
    assert detail["preferences"]["shared_device"] and detail["consent"]["purpose"] == "Report to police"
    assert detail["evidence_log"]["intact"]
    # Preferences are encrypted at rest.
    from safepay.app.main import store
    assert note not in json.dumps(store.q("SELECT * FROM prefs"))
    assert "gym" not in json.dumps(store.q("SELECT * FROM pairs"))   # relationship state too
    # Pack: created, retrievable, verifiable; never contains the safe-contact note.
    made = c.post(f"/v1/cases/{case_id}/evidence-pack").json()
    pack = c.get(f"/v1/packs/{made['pack_id']}").json()
    assert pack["sha256"] == made["sha256"] and note not in json.dumps(pack)
    assert len(pack["pack"]["timeline"]) == 8                         # all 8, including those before the case opened
    v = c.post("/v1/packs/verify", json=pack).json()
    assert v["valid"] and v["issued_by_this_service"] and v["evidence_log_still_intact"]
    pack["pack"]["timeline"][0]["message"] = "edited"
    assert not c.post("/v1/packs/verify", json=pack).json()["valid"]
    # Withdrawn consent stops new packs.
    c.post("/demo/safety/sam/consent", json={"sender": "riley", "grant": False})
    assert c.post(f"/v1/cases/{case_id}/evidence-pack").status_code == 403
