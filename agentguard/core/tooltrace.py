"""Guard 3: spot dangerous *sequences* of tool calls.

Guards 1 and 2 read text. This one ignores text entirely and watches behaviour,
so it still fires when an attack is phrased in a way the text models have never
seen. The signal that matters is ordering: reading untrusted content, then
pulling bulk records, then sending something outward.

Sessions here are operational telemetry (tool names, amounts, timings), not
natural language, so they are generated locally.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

READ_TOOLS = {"get_balance", "recent_transactions"}
BULK_TOOLS = {"lookup_customer", "export_all_customers"}
EGRESS_TOOLS = {"send_email", "transfer"}
UNTRUSTED_TOOLS = {"read_document"}
ALL_TOOLS = sorted(READ_TOOLS | BULK_TOOLS | EGRESS_TOOLS | UNTRUSTED_TOOLS)


@dataclass
class ToolCall:
    tool: str
    t: float                      # seconds since session start
    amount: float = 0.0           # transfers only
    payee_known: bool = True      # transfers only
    rows_returned: int = 0        # reads only


@dataclass
class Session:
    calls: list[ToolCall] = field(default_factory=list)
    hist_mean: float = 300.0      # customer's usual transfer size
    hist_max: float = 900.0

    def add(self, tool: str, **kw) -> "Session":
        gap = kw.pop("gap", 4.0)
        t = self.calls[-1].t + gap if self.calls else 0.0
        self.calls.append(ToolCall(tool=tool, t=t, **kw))
        return self


@dataclass
class TraceFeatures:
    n_calls: int
    distinct_tools: int
    bulk_calls: int
    rows_returned: int
    untrusted_read: int
    egress_after_untrusted: int
    egress_after_bulk: int
    transfers: int
    max_amount_ratio: float       # amount / customer's usual max
    unknown_payee_transfers: int
    calls_per_minute: float
    bulk_ratio: float

    def vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def names() -> list[str]:
        return list(TraceFeatures.__annotations__.keys())


def extract(s: Session) -> TraceFeatures:
    calls = s.calls
    n = len(calls)
    seen_untrusted = seen_bulk = False
    egress_after_untrusted = egress_after_bulk = 0
    for c in calls:
        if c.tool in UNTRUSTED_TOOLS:
            seen_untrusted = True
        if c.tool in BULK_TOOLS:
            seen_bulk = True
        if c.tool in EGRESS_TOOLS:
            egress_after_untrusted += int(seen_untrusted)
            egress_after_bulk += int(seen_bulk)

    transfers = [c for c in calls if c.tool == "transfer"]
    span = (calls[-1].t - calls[0].t) if n > 1 else 1.0
    bulk = sum(1 for c in calls if c.tool in BULK_TOOLS)
    return TraceFeatures(
        n_calls=n,
        distinct_tools=len({c.tool for c in calls}),
        bulk_calls=bulk,
        rows_returned=sum(c.rows_returned for c in calls),
        untrusted_read=sum(1 for c in calls if c.tool in UNTRUSTED_TOOLS),
        egress_after_untrusted=egress_after_untrusted,
        egress_after_bulk=egress_after_bulk,
        transfers=len(transfers),
        max_amount_ratio=max((c.amount / max(s.hist_max, 1.0) for c in transfers), default=0.0),
        unknown_payee_transfers=sum(1 for c in transfers if not c.payee_known),
        calls_per_minute=n / max(span / 60.0, 1 / 60.0),
        bulk_ratio=bulk / max(n, 1),
    )


# ---------------------------------------------------------------------------
# Synthetic session generator
# ---------------------------------------------------------------------------
BENIGN_SHAPES = [
    ("check", 0.22), ("pay", 0.20), ("review", 0.16), ("mixed", 0.12),
    # Hard negatives: these look like attacks and must stay labelled benign.
    ("support_case", 0.12),   # repeated lookups, then an email to that customer
    ("big_payment", 0.07),    # house deposit, far above the usual ceiling
    ("reconcile", 0.06),      # standalone bulk export by the account holder
    ("new_payee", 0.05),      # first payment to someone not in the payee list
]


def _benign(rng: random.Random) -> Session:
    s = Session(hist_mean=rng.uniform(120, 600), hist_max=rng.uniform(700, 1500))
    shapes, weights = zip(*BENIGN_SHAPES)
    shape = rng.choices(shapes, weights=weights, k=1)[0]
    if shape == "support_case":
        for _ in range(rng.randint(2, 6)):
            s.add("lookup_customer", gap=rng.uniform(3, 20), rows_returned=rng.randint(1, 4))
        if rng.random() < 0.7:
            s.add("send_email", gap=rng.uniform(6, 50))
        return s
    if shape == "big_payment":
        s.add("get_balance", rows_returned=1)
        s.add("transfer", gap=rng.uniform(15, 120),
              amount=s.hist_max * rng.uniform(3, 15),
              payee_known=rng.random() < 0.6)
        return s
    if shape == "reconcile":
        s.add("export_all_customers", rows_returned=rng.randint(20, 120))
        if rng.random() < 0.3:
            s.add("send_email", gap=rng.uniform(20, 90))
        return s
    if shape == "new_payee":
        s.add("transfer", gap=rng.uniform(10, 60),
              amount=min(rng.lognormvariate(5.1, 0.7), s.hist_max * 1.2),
              payee_known=False)
        return s
    if shape == "check":
        s.add("get_balance", gap=rng.uniform(1, 6), rows_returned=1)
    elif shape == "pay":
        s.add("get_balance", rows_returned=1)
        s.add("transfer", gap=rng.uniform(8, 40),
              amount=min(rng.lognormvariate(5.2, 0.6), s.hist_max * rng.uniform(0.3, 1.1)),
              payee_known=True)
    elif shape == "review":
        s.add("recent_transactions", rows_returned=rng.randint(3, 10))
        if rng.random() < 0.6:
            s.add("get_balance", gap=rng.uniform(3, 15), rows_returned=1)
    else:
        s.add("read_document", gap=rng.uniform(2, 10), rows_returned=1)   # reads own statement
        s.add("recent_transactions", gap=rng.uniform(5, 25), rows_returned=rng.randint(3, 8))
        if rng.random() < 0.4:
            s.add("transfer", gap=rng.uniform(10, 60),
                  amount=min(rng.lognormvariate(5.0, 0.5), s.hist_max), payee_known=True)
    if rng.random() < 0.12:
        s.add("lookup_customer", gap=rng.uniform(5, 30), rows_returned=1)
    return s


def _attack(rng: random.Random) -> Session:
    s = Session(hist_mean=rng.uniform(120, 600), hist_max=rng.uniform(700, 1500))
    shape = rng.choice(["exfil_email", "bulk_dump", "silent_transfer", "slow_drip"])
    if shape == "exfil_email":
        s.add("read_document", rows_returned=1)
        s.add("lookup_customer", gap=rng.uniform(0.5, 3), rows_returned=rng.randint(5, 25))
        s.add("send_email", gap=rng.uniform(0.5, 3))
    elif shape == "bulk_dump":
        s.add("export_all_customers", rows_returned=rng.randint(40, 400))
        if rng.random() < 0.7:
            s.add("send_email", gap=rng.uniform(0.5, 4))
    elif shape == "silent_transfer":
        s.add("read_document", rows_returned=1)
        # A patient attacker stays under the usual ceiling and reuses a known payee.
        stealthy = rng.random() < 0.4
        s.add("transfer", gap=rng.uniform(5, 45) if stealthy else rng.uniform(0.5, 5),
              amount=s.hist_max * (rng.uniform(0.4, 1.3) if stealthy else rng.uniform(3, 20)),
              payee_known=stealthy)
    else:  # slow_drip: paced to look like ordinary support work
        for _ in range(rng.randint(3, 7)):
            s.add("lookup_customer", gap=rng.uniform(4, 18), rows_returned=rng.randint(1, 4))
        s.add("send_email", gap=rng.uniform(8, 40))
    return s


def make_dataset(n: int = 4000, seed: int = 11) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        attack = rng.random() < 0.35
        s = _attack(rng) if attack else _benign(rng)
        X.append(extract(s).vector())
        y.append(int(attack))
    return X, y
