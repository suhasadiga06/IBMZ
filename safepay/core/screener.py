"""The live screening service: one ISO 20022 payment in, one decision out.

Per payment (all in-process, no network hop):
  pseudonymise accounts -> score message -> score joined recent messages ->
  update relationship state -> score relationship -> policy -> persist ->
  forward the payment (with the message withheld if needed)
"""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime

import numpy as np

from . import iso20022
from .engine import Engine, informative
from .policy import Thresholds, decide
from .privacy import Privacy
from .state import PairState
from .store import Store


class Screener:
    def __init__(self, text, rel_bundle: dict, store: Store, privacy: Privacy, on_new_case=None, on_escalate=None):
        self.engine = Engine(text, rel_bundle["model"])
        self.t = Thresholds(harm=text.thresholds["harm"], threat=text.thresholds["threat"],
                            case=rel_bundle["threshold"])
        self.store, self.privacy = store, privacy
        self.on_new_case = on_new_case
        self.on_escalate = on_escalate
        self.model_version = rel_bundle.get("version", "dev")
        self.persist = True
        for row in store.load_states():
            self.engine.states[(row["sender"], row["recipient"])] = PairState.from_json(row["state"])
        for h in store.q("SELECT * FROM hides"):
            self.engine.hide(h["recipient"], h["sender"], h["ts"])

    # -- helpers -------------------------------------------------------------------
    def tokens(self, p: iso20022.Payment) -> tuple[str, str]:
        return self.privacy.token(p.debtor_account), self.privacy.token(p.creditor_account)

    def _peek(self, when: datetime, s: str, r: str, amount: float, ref: str):
        """Score as if the payment were sent, without changing any state (pre-submit check)."""
        st = self.engine.states.get((s, r)) or PairState()
        tmp = PairState.from_json(st.to_json())
        ts = when.timestamp()
        harm, threat = self.engine.score_text(ref) if ref else (0.0, 0.0)
        joined = tmp.split_text(ts, ref)
        sh, sth = self.engine.score_text(joined) if joined else (0.0, 0.0)
        tmp.update(ts, when.hour, amount, ref, harm, threat, sh, sth)
        score = float(self.engine.rel.predict_proba(np.asarray([tmp.features(ts)], dtype=np.float32))[0, 1])
        return harm, threat, joined, sh, sth, score, tmp

    def hide_sender(self, recipient_acct: str, sender_acct: str) -> None:
        r, s = self.privacy.token(recipient_acct), self.privacy.token(sender_acct)
        now = time.time()
        self.store.hide(r, s, now)
        self.engine.hide(r, s, now)
        self.store.audit("recipient", "hide_sender", None, f"{r} hid {s}")

    # -- main entry ------------------------------------------------------------------
    def screen(self, p: iso20022.Payment, presubmit: bool = False) -> dict:
        t0 = time.perf_counter()
        s, r = self.tokens(p)
        case = self.store.case_for(s, r)
        if presubmit:
            harm, threat, joined, sh, sth, score, st = self._peek(p.created, s, r, p.amount, p.reference)
        else:
            obs = self.engine.observe(p.created, s, r, p.amount, p.reference)
            harm, threat, joined, sh, sth, score, st = (obs.harm, obs.threat, obs.split_text, obs.split_harm,
                                                        obs.split_threat, obs.rel_score, obs.state)
        t_models = time.perf_counter()
        hidden_by_recipient = bool(st.hidden_at) and p.created.timestamp() >= st.hidden_at
        d = decide(self.t, harm, threat, sh, sth, score, case is not None, hidden_by_recipient, presubmit,
                   n_payments=st.n_tx)

        case_id = case["id"] if case else None
        new_case = False
        if not presubmit:
            if d.case in ("open", "update") or (case and d.message == "HIDE"):
                new_case = case is None
                case_id = self.store.open_or_update(s, r, p.created.timestamp(), d.priority or (case or {}).get("priority", "P3"),
                                                    score, d.reasons)
                if new_case:
                    # The messages just before the case opened are part of the pattern.
                    for ts, ref in st.buffer[:-1] if p.reference else st.buffer:
                        h, t = self.engine.score_text(ref)
                        self.store.add_evidence(case_id, ts, 0.0, self.privacy.seal(ref, f"case:{case_id}"), h, t, False)
                elif d.priority == "P1" and case.get("priority") != "P1" and self.on_escalate:
                    self.on_escalate(case_id)
            if case_id and p.reference:
                ctx = f"case:{case_id}"
                self.store.add_evidence(case_id, p.created.timestamp(), p.amount,
                                        self.privacy.seal(p.reference, ctx), harm, threat, d.message == "HIDE")
            if self.persist:
                self.store.save_state(s, r, st.to_json())
            if new_case and self.on_new_case:
                self.on_new_case(case_id)
        t_end = time.perf_counter()

        detail = None
        # Word-level evidence only when the words themselves were the reason.
        if d.message == "HIDE" and informative(p.reference) and (harm >= self.t.harm or threat >= self.t.threat):
            m = self.engine.text.score(p.reference)
            detail = {"evidence": m.evidence, "obfuscation": m.obfuscation}
        forwarded = iso20022.with_reference(p, iso20022.WITHHELD) if d.message == "HIDE" else iso20022.build_pacs008(p)
        return {
            "end_to_end_id": p.end_to_end_id,
            "decision": asdict(d),
            "scores": {"message_scored": informative(p.reference),
                       "message_harm": round(harm, 4), "message_threat": round(threat, 4),
                       "joined_recent": joined, "joined_harm": round(sh, 4), "joined_threat": round(sth, 4),
                       "relationship": round(score, 4)},
            "thresholds": asdict(self.t),
            "explain": detail,
            "relationship": {"payments": st.n_tx, "penny_share": round(st.n_micro / max(st.n_tx, 1), 3),
                             "night_share": round(st.n_night / max(st.n_tx, 1), 3), "replies": st.n_reply,
                             "harmful_messages": st.n_harmful, "contact_after_hide": st.n_after_hide,
                             "escalation": round(st.cusum, 3)},
            "case_id": case_id, "new_case": new_case,
            "sender_token": s, "recipient_token": r,
            "latency_ms": {"models": round(1000 * (t_models - t0), 3), "total": round(1000 * (t_end - t0), 3)},
            "model_version": self.model_version,
            "forwarded_pacs008": forwarded if not presubmit else None,
        }
