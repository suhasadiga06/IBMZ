"""What the bank does with each payment. Mirrors current practice at banks
that tackle payment-based abuse, plus two safety rules from victim-support
guidance:

  * Money is never blocked automatically. It may be child support or rent the
    victim depends on. Only the *message* is withheld.
  * The sender is never told they have been flagged. Warning an abuser that
    they are being watched can escalate danger, so once a relationship is
    under review, harmful messages are withheld silently.

Actions per payment:
  message   SHOW | HIDE            (hidden text is replaced in the forwarded pacs.008)
  case      none | open | update    (with priority P1..P3 for the review queue)
  nudge     only at the sender's bank, only before sending, only for a first
            offence by someone not under review ("please keep messages respectful")
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Thresholds:
    harm: float
    threat: float
    case: float


@dataclass
class Decision:
    deliver_funds: bool = True
    message: str = "SHOW"
    case: str = "none"
    priority: str = ""
    nudge_sender: bool = False
    reasons: list[str] = field(default_factory=list)


MIN_EVIDENCE = 3   # payments before a case may open on behaviour alone
# After a customer hides someone, a brand-new sender who has never been paid
# back gets their messages hidden by default for this long (one tap to view).
# An abuser who switches accounts is a stranger to the models for his first few
# payments; this extends the victim's own choice to his next account.
PROTECT_DAYS = 14


def protect_after_hide(recipient_ctx: list[float], replies: int) -> bool:
    hid_someone, days_since, _ = recipient_ctx
    return hid_someone == 1.0 and days_since <= PROTECT_DAYS and replies == 0


def content_flag(t: Thresholds, harm: float, threat: float, split_harm: float, split_threat: float) -> bool:
    return harm >= t.harm or threat >= t.threat or split_harm >= t.harm or split_threat >= t.threat


def opens_case(t: Thresholds, rel_score: float, n_payments: int, content: bool) -> bool:
    """Same rule in evaluation and in the live service."""
    return rel_score >= t.case and (n_payments >= MIN_EVIDENCE or content)


def decide(t: Thresholds, harm: float, threat: float, split_harm: float, split_threat: float,
           rel_score: float, under_review: bool, recipient_hid_sender: bool, presubmit: bool = False,
           n_payments: int = MIN_EVIDENCE, new_sender_after_hide: bool = False) -> Decision:
    d = Decision()
    if threat >= t.threat:
        d.reasons.append(f"message reads as a threat ({threat:.2f})")
    if harm >= t.harm:
        d.reasons.append(f"abusive message ({harm:.2f})")
    if split_threat >= t.threat or split_harm >= t.harm:
        d.reasons.append(f"joined with recent payments, the messages form a {'threat' if split_threat >= t.threat else 'harmful message'}")
    content = content_flag(t, harm, threat, split_harm, split_threat)
    case_now = opens_case(t, rel_score, n_payments, content) or (under_review and rel_score >= t.case)
    if case_now:
        d.reasons.append(f"relationship pattern looks abusive ({rel_score:.2f})")
    if recipient_hid_sender:
        d.reasons.append("recipient chose to hide messages from this sender")
    if new_sender_after_hide and not d.reasons:
        d.reasons.append("new sender soon after the recipient hid someone else")

    message_bad = bool(d.reasons)
    if message_bad:
        d.message = "HIDE"

    if case_now:
        d.case = "update" if under_review else "open"
        severe = threat >= t.threat or split_threat >= t.threat
        d.priority = "P1" if severe else ("P2" if rel_score >= (1 + t.case) / 2 else "P3")
    elif under_review and message_bad:
        d.case = "update"

    if presubmit and (harm >= t.harm or threat >= t.threat) and not under_review and rel_score < t.case:
        d.nudge_sender = True
    return d
