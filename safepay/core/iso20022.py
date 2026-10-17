"""ISO 20022 payment messages: the format banks actually exchange.

SafePay reads the two messages a bank sees for a credit transfer:

  pain.001  customer -> own bank (the sender's bank screens before sending)
  pacs.008  bank -> bank         (the recipient's bank screens on arrival, so a
                                  victim is protected even if the abuser banks
                                  somewhere else)

The free-text remittance field, RmtInf/Ustrd (max 140 characters), is where
abusive messages travel. For a payment whose message is withheld, we return
the same pacs.008 with only that field replaced, so the money still arrives.
"""
from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

NS = {
    "pacs.008": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08",
    "pain.001": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09",
}
USTRD_MAX = 140
WITHHELD = "Message withheld by your bank for your safety"


@dataclass
class Payment:
    msg_id: str
    end_to_end_id: str
    created: datetime
    amount: float
    currency: str
    debtor_name: str
    debtor_account: str
    creditor_name: str
    creditor_account: str
    reference: str
    kind: str = "pacs.008"


def _acct(parent: ET.Element, ns: str, tag: str) -> str:
    node = parent.find(f"{{{ns}}}{tag}/{{{ns}}}Id")
    if node is None:
        return ""
    iban = node.find(f"{{{ns}}}IBAN")
    if iban is not None and iban.text:
        return iban.text.strip()
    other = node.find(f"{{{ns}}}Othr/{{{ns}}}Id")
    return other.text.strip() if other is not None and other.text else ""


def _text(parent: ET.Element | None, path: str, ns: str, default: str = "") -> str:
    if parent is None:
        return default
    node = parent.find("/".join(f"{{{ns}}}{p}" for p in path.split("/")))
    return node.text.strip() if node is not None and node.text else default


def parse(xml: str | bytes) -> list[Payment]:
    """Parse a pacs.008 or pain.001 document into one Payment per transaction."""
    root = ET.fromstring(xml)
    ns = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
    kind = next((k for k, v in NS.items() if v == ns), None)
    if kind is None:
        raise ValueError(f"unsupported message namespace: {ns or '(none)'}")
    body = root[0]
    created = _text(body, "GrpHdr/CreDtTm", ns) or datetime.now(timezone.utc).isoformat()
    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    msg_id = _text(body, "GrpHdr/MsgId", ns)
    out: list[Payment] = []

    if kind == "pacs.008":
        for tx in body.findall(f"{{{ns}}}CdtTrfTxInf"):
            amt = tx.find(f"{{{ns}}}IntrBkSttlmAmt")
            out.append(Payment(
                msg_id=msg_id, end_to_end_id=_text(tx, "PmtId/EndToEndId", ns), created=created_dt,
                amount=float(amt.text), currency=amt.get("Ccy", ""),
                debtor_name=_text(tx, "Dbtr/Nm", ns), debtor_account=_acct(tx, ns, "DbtrAcct"),
                creditor_name=_text(tx, "Cdtr/Nm", ns), creditor_account=_acct(tx, ns, "CdtrAcct"),
                reference=" ".join(n.text or "" for n in tx.findall(f"{{{ns}}}RmtInf/{{{ns}}}Ustrd")).strip(),
                kind=kind))
    else:
        for pmt in body.findall(f"{{{ns}}}PmtInf"):
            debtor, debtor_acct = _text(pmt, "Dbtr/Nm", ns), _acct(pmt, ns, "DbtrAcct")
            when = _text(pmt, "ReqdExctnDt/DtTm", ns)
            for tx in pmt.findall(f"{{{ns}}}CdtTrfTxInf"):
                amt = tx.find(f"{{{ns}}}Amt/{{{ns}}}InstdAmt")
                out.append(Payment(
                    msg_id=msg_id, end_to_end_id=_text(tx, "PmtId/EndToEndId", ns),
                    created=datetime.fromisoformat(when.replace("Z", "+00:00")) if when else created_dt,
                    amount=float(amt.text), currency=amt.get("Ccy", ""),
                    debtor_name=debtor, debtor_account=debtor_acct,
                    creditor_name=_text(tx, "Cdtr/Nm", ns), creditor_account=_acct(tx, ns, "CdtrAcct"),
                    reference=" ".join(n.text or "" for n in tx.findall(f"{{{ns}}}RmtInf/{{{ns}}}Ustrd")).strip(),
                    kind=kind))
    return out


def build_pacs008(p: Payment) -> str:
    """Serialise one payment as a minimal, schema-shaped pacs.008.001.08."""
    ns = NS["pacs.008"]
    ET.register_namespace("", ns)
    q = lambda t: f"{{{ns}}}{t}"
    doc = ET.Element(q("Document"))
    body = ET.SubElement(doc, q("FIToFICstmrCdtTrf"))
    hdr = ET.SubElement(body, q("GrpHdr"))
    ET.SubElement(hdr, q("MsgId")).text = p.msg_id
    ET.SubElement(hdr, q("CreDtTm")).text = p.created.isoformat(timespec="seconds")
    ET.SubElement(hdr, q("NbOfTxs")).text = "1"
    sttl = ET.SubElement(ET.SubElement(hdr, q("SttlmInf")), q("SttlmMtd"))
    sttl.text = "CLRG"
    tx = ET.SubElement(body, q("CdtTrfTxInf"))
    ET.SubElement(ET.SubElement(tx, q("PmtId")), q("EndToEndId")).text = p.end_to_end_id
    amt = ET.SubElement(tx, q("IntrBkSttlmAmt"), Ccy=p.currency)
    amt.text = f"{p.amount:.2f}"
    ET.SubElement(tx, q("ChrgBr")).text = "SLEV"
    for role, name, acct in (("Dbtr", p.debtor_name, p.debtor_account), ("Cdtr", p.creditor_name, p.creditor_account)):
        ET.SubElement(ET.SubElement(tx, q(role)), q("Nm")).text = name
        idn = ET.SubElement(ET.SubElement(ET.SubElement(tx, q(role + "Acct")), q("Id")), q("Othr"))
        ET.SubElement(idn, q("Id")).text = acct
    if p.reference:
        ET.SubElement(ET.SubElement(tx, q("RmtInf")), q("Ustrd")).text = p.reference[:USTRD_MAX]
    return ET.tostring(doc, encoding="unicode", xml_declaration=True)


def with_reference(p: Payment, reference: str) -> str:
    """The forwarded message: identical payment, different remittance text."""
    return build_pacs008(Payment(**{**p.__dict__, "reference": reference}))


def new_payment(sender: str, sender_acct: str, recipient: str, recipient_acct: str,
                amount: float, reference: str, when: datetime | None = None, currency: str = "AUD") -> Payment:
    when = when or datetime.now(timezone.utc)
    uid = uuid.uuid4().hex[:16].upper()
    return Payment(msg_id=f"MSG{uid}", end_to_end_id=f"E2E{uid}", created=when, amount=round(amount, 2),
                   currency=currency, debtor_name=sender, debtor_account=sender_acct,
                   creditor_name=recipient, creditor_account=recipient_acct,
                   reference=reference[:USTRD_MAX])
