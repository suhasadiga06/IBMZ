"""A synthetic payments world with abusive and innocent relationships.

No public dataset of real abusive payments exists, and one should not: it
would expose victims. So the *timelines* are generated here, following the
patterns banks have published (penny payments, messages split across
transfers, one-way contact that continues after the victim blocks the sender,
abusive references on otherwise legitimate child-support payments).

What keeps the evaluation honest:
  * Message text is real: Civil Comments train split for the training world,
    test split for the test world.
  * Phrases written for this simulation (payment-style references, banter,
    coercive messages) are split in thirds: one each for the train, validation
    and test worlds, and lightly varied so no model can key on exact strings.
  * Innocent relationships include deliberately confusing cases, and abusive
    ones include quiet, low-toxicity cases, so behaviour alone cannot win.
  * The relationship model never sees text, only the message model's scores.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
START = datetime(2026, 5, 1, tzinfo=timezone.utc)
DAYS = 120

FIRST = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Ethan", "Isla", "Leo", "Ruby", "Jack", "Aria", "Kai", "Priya",
         "Arjun", "Mei", "Hana", "Omar", "Layla", "Diego", "Sofia", "Tane", "Aroha", "Sam", "Alex", "Jordan",
         "Riley", "Chloe", "Lucas", "Ella", "Mason", "Ivy", "Felix", "Nina", "Ravi", "Anika", "Yusuf"]
LAST = ["Smith", "Nguyen", "Patel", "Brown", "Wilson", "Taylor", "Kaur", "Singh", "Chen", "Williams",
        "Martin", "Lee", "Walker", "Hall", "Young", "King", "Wright", "Scott", "Green", "Baker", "Kim"]

# Phrases written for the simulation. Each list is split in thirds: one each
# for the training, validation and test worlds.
PAYMENT_REFS = [
    "rent", "groceries", "pizza", "thanks for dinner", "bday present", "uber split", "electricity bill",
    "school fees", "petrol", "concert ticket", "paying you back", "happy birthday x", "coffee", "lunch",
    "netflix share", "gas bill", "footy tickets", "dinner friday", "drinks", "movie night", "water bill",
    "gift for mum", "holiday deposit", "taxi", "brunch", "for the bbq", "phone bill", "internet",
    "flowers", "parking", "trip money", "rego", "sushi", "kebab lol", "wedding gift", "baby shower",
    "for the kids", "groceries split", "beers", "thank you!", "cheers mate", "half of the bill", "",
]
# Friendly rudeness, from mild to harsh, so every third of the list contains
# messages a toxicity model will score as abusive even though they are not.
BANTER = [
    "you owe me loser", "pay up you muppet", "beer money idiot", "killing it at work congrats",
    "dead funny night lol", "for being a legend", "ur the worst at pool haha", "shut up and take my money",
    "die hard tickets", "you absolute weapon", "stop being cheap lol", "for the stupid cab", "you nerd",
    "for losing the bet sucker", "bloody expensive dinner", "ur fired as chef lol", "hate you but here",
    "you clown, here's your cut", "dinner, you pig", "gone mad with pizza",
    "you're an idiot for picking that place", "stupid parking fine", "damn that was a lot of beer",
    "fat burger fund", "you dumbass, here's the rent", "crap movie but thanks", "for the terrible pizza lol",
    "lazy git, your share", "you cheeky sod", "go to hell, here's your money", "you ugly mug lol",
    "kill it at the interview", "die laughing at that show", "stupid taxi was so dear", "your crappy uber",
    "idiot tax for forgetting your wallet", "screw you and your fancy wine", "silly goose, bus money",
    "you useless lump, groceries", "that bill was huge wtf", "moron money for the bet", "dumb concert ticket",
    "you're dead to me after that game lol", "freaking expensive brunch", "loser buys drinks",
]
COERCIVE = [
    "unblock me", "answer your phone", "why are you ignoring me", "I know you're at your mum's",
    "saw you at the shops today", "you can't hide from me", "pick up", "we need to talk now",
    "you'll regret this", "I'm outside", "I still have your keys", "call me back or else",
    "nice dress today", "say hi to your new friend", "I know where the kids go to school",
    "you belong with me", "nobody else will have you", "check your messages", "I'm watching",
    "don't make me come over", "who were you with last night", "you're nothing without me",
    "this isn't over", "you can't keep the kids from me", "I'll find you", "answer me",
    "I saw your car", "reply or I'll call your work", "you think you can leave me", "open the door",
]
LOVING = ["love you xx", "for my favourite person", "dinner date", "miss you", "for the kids xx",
          "groceries babe", "date night", "you're the best", "anniversary dinner", "sorry about today x"]
FIRM = ["please pay back the $200", "still waiting on the loan", "reminder: money for the car",
        "can you sort the rent you owe", "final reminder about the bond", "please repay when you can",
        "you said you'd pay me last week", "need that money back for the bills", "your half of the phone bill",
        "it's been a month, pay me back", "still owe me for the tickets", "can you transfer the $50 back"]
CHILD_SUPPORT = ["child support", "for the kids", "kids money", "school shoes", "maintenance", ""]


@dataclass
class Tx:
    ts: datetime
    sender: str
    recipient: str
    amount: float
    reference: str
    rel_id: int


@dataclass
class Relationship:
    rel_id: int
    sender: str
    recipient: str
    kind: str
    label: int
    first_abuse: datetime | None = None   # first abusive payment, for detection delay


@dataclass
class World:
    customers: dict[str, str]                  # account -> name
    relationships: list[Relationship]
    txs: list[Tx]
    hides: list[tuple[datetime, str, str]]     # (when, recipient, sender) "hide messages from"
    meta: dict = field(default_factory=dict)


class TextPools:
    """Real comments from Civil Comments, by split."""

    def __init__(self, split: str, rng: random.Random):
        path = DATA / f"cc_{split}.csv"
        df = pd.read_csv(path).dropna(subset=["text"])
        df = df[df.text.str.len() <= 140]
        self.toxic = df[df.toxicity >= 0.5].text.tolist()
        self.threat = df[(df.threat >= 0.5) & (df.text.str.split().str.len() >= 5)].text.tolist()
        self.chatty = df[(df.toxicity < 0.1) & (df.text.str.len() <= 60)].text.tolist()
        # Each world gets its own third of every hand-written list, so validation
        # measures generalisation to unseen wording just as the test does.
        third = {"train": 0, "val": 1, "test": 2}[split]
        pick = lambda xs: xs[third::3]
        self.refs, self.banter, self.coercive = pick(PAYMENT_REFS), pick(BANTER), pick(COERCIVE)
        self.loving, self.firm, self.support = pick(LOVING), pick(FIRM), CHILD_SUPPORT
        self.rng = rng
        self._written = {id(x) for x in (self.refs, self.banter, self.coercive, self.loving, self.firm)}

    def one(self, pool: list[str]) -> str:
        if not pool:
            return ""
        text = self.rng.choice(pool)
        return vary(text, self.rng) if id(pool) in self._written else text


def vary(text: str, rng: random.Random) -> str:
    """Small, realistic variation so models cannot key on a handful of exact strings."""
    if not text:
        return text
    r = rng.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.3:
        text = text.capitalize()
    r = rng.random()
    if r < 0.15:
        text += rng.choice(["!", "!!", "?", " pls", " now", " x", " thanks", " ok"])
    elif r < 0.25:
        text = rng.choice(["hey ", "hi ", "so ", "again ", "re: "]) + text
    return text[:140]


class Builder:
    def __init__(self, split: str, seed: int):
        self.rng = random.Random(seed)
        self.text = TextPools(split, self.rng)
        self.customers: dict[str, str] = {}
        self.rels: list[Relationship] = []
        self.txs: list[Tx] = []
        self.hides: list[tuple[datetime, str, str]] = []

    # -- helpers --------------------------------------------------------------
    def person(self) -> str:
        acct = f"{self.rng.randint(10, 99)}{self.rng.randint(1000, 9999)}-{self.rng.randint(10**7, 10**8 - 1)}"
        self.customers[acct] = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
        return acct

    def when(self, day: float, night_bias: float = 0.1) -> datetime:
        r = self.rng
        hour = r.uniform(0, 5) if r.random() < night_bias else r.triangular(7, 23, 18)
        return START + timedelta(days=int(day), hours=hour, minutes=r.uniform(0, 59))

    def pay(self, rel: Relationship, ts: datetime, amount: float, ref: str, reverse: bool = False) -> None:
        s, t = (rel.recipient, rel.sender) if reverse else (rel.sender, rel.recipient)
        self.txs.append(Tx(ts, s, t, round(max(amount, 0.01), 2), ref[:140], rel.rel_id))

    def rel(self, kind: str, label: int) -> Relationship:
        r = Relationship(len(self.rels), self.person(), self.person(), kind, label)
        self.rels.append(r)
        return r

    def micro(self) -> float:
        return self.rng.choice([0.01, 0.01, 0.01, 0.05, 0.10, 0.5, 1.0])

    # -- innocent relationships ----------------------------------------------
    def friends(self):
        r, t = self.rel("friends", 0), self.text
        d0 = self.rng.uniform(0, DAYS - 10)
        for _ in range(self.rng.randint(2, 14)):
            d = min(d0 + self.rng.expovariate(1 / 8), DAYS - 1)
            roll = self.rng.random()
            ref = t.one(t.banter) if roll < 0.12 else t.one(t.chatty) if roll < 0.3 else t.one(t.refs)
            self.pay(r, self.when(d, 0.08), self.rng.lognormvariate(3.3, 0.7), ref,
                     reverse=self.rng.random() < 0.45)

    def couple(self):
        r, t = self.rel("couple_or_family", 0), self.text
        for _ in range(self.rng.randint(6, 30)):
            d = self.rng.uniform(0, DAYS - 1)
            ref = t.one(t.loving) if self.rng.random() < 0.4 else t.one(t.refs)
            self.pay(r, self.when(d, 0.15), self.rng.lognormvariate(3.5, 0.9), ref,
                     reverse=self.rng.random() < 0.5)

    def allowance(self):
        r = self.rel("parent_allowance", 0)
        amt, d = self.rng.choice([10, 20, 25, 30, 50]), self.rng.uniform(0, 7)
        while d < DAYS:
            self.pay(r, self.when(d, 0.02), amt, self.rng.choice(["pocket money", "allowance", "", "lunch money"]))
            d += 7

    def rent(self):
        r = self.rel("rent", 0)
        amt, d = round(self.rng.uniform(900, 2600), -1), self.rng.uniform(0, 28)
        while d < DAYS:
            self.pay(r, self.when(d, 0.02), amt, f"rent {(START + timedelta(days=d)).strftime('%b').lower()}")
            d += 30

    def child_support(self):
        r = self.rel("child_support", 0)
        amt, step, d = round(self.rng.uniform(150, 700), -1), self.rng.choice([14, 30]), self.rng.uniform(0, 14)
        while d < DAYS:
            self.pay(r, self.when(d, 0.05), amt, self.text.one(self.text.support))
            d += step

    def debt_reminder(self):
        r, t = self.rel("debt_reminder", 0), self.text
        d = self.rng.uniform(0, 60)
        for _ in range(self.rng.randint(2, 5)):
            self.pay(r, self.when(d, 0.05), self.micro(), t.one(t.firm))
            d += self.rng.uniform(5, 20)
        if self.rng.random() < 0.7:
            self.pay(r, self.when(min(d, DAYS - 1)), self.rng.uniform(100, 400), "paid back sorry", reverse=True)

    def verification(self):
        r = self.rel("account_verification", 0)
        d = self.rng.uniform(0, DAYS - 1)
        for _ in range(self.rng.randint(1, 2)):
            self.pay(r, self.when(d), 0.01, self.rng.choice(["test", "verification", "check", "1c test"]))

    def night_worker(self):
        r, t = self.rel("night_shift_friends", 0), self.text
        for _ in range(self.rng.randint(4, 16)):
            self.pay(r, self.when(self.rng.uniform(0, DAYS - 1), 0.8), self.rng.lognormvariate(3.0, 0.6),
                     t.one(t.refs), reverse=self.rng.random() < 0.4)

    def marketplace(self):
        r = self.rel("marketplace", 0)
        self.pay(r, self.when(self.rng.uniform(0, DAYS - 1)), self.rng.uniform(15, 400),
                 self.rng.choice(["bike", "couch", "ps5", "pram", "desk", "tickets", "textbook", ""]))

    # -- abusive relationships -------------------------------------------------
    def _abuse_ref(self, harm_level: float) -> str:
        t, roll = self.text, self.rng.random()
        if roll < harm_level:
            return t.one(t.toxic)
        if roll < harm_level + 0.35:
            return t.one(t.coercive)
        return t.one(t.refs) if self.rng.random() < 0.5 else ""

    def _campaign(self, r: Relationship, d0: float, n: int, harm: float, micro_p: float, night: float,
                  replies: float, hide_p: float, split_threats: int = 0):
        d, length = d0, self.rng.uniform(7, 60)
        times = sorted(d0 + length * (self.rng.random() ** 0.7) for _ in range(n))  # denser later: escalation
        hide_after = int(n * self.rng.uniform(0.2, 0.6)) if self.rng.random() < hide_p else None
        splits_at = set(self.rng.sample(range(n), k=min(split_threats, n)))
        for i, d in enumerate(times):
            d = min(d, DAYS - 1)
            ts = self.when(d, night)
            amount = self.micro() if self.rng.random() < micro_p else self.rng.lognormvariate(3.0, 0.8)
            if i in splits_at and self.text.threat:
                words = self.text.one(self.text.threat).split()
                k = 0
                while k < len(words):
                    step = self.rng.randint(1, 3)
                    self.pay(r, ts, 0.01, " ".join(words[k:k + step]))
                    ts += timedelta(seconds=self.rng.uniform(10, 120))
                    k += step
            else:
                self.pay(r, ts, amount, self._abuse_ref(harm * (0.6 + 0.8 * i / max(n - 1, 1))))
            if self.rng.random() < replies * (1 - i / n):
                self.pay(r, ts + timedelta(hours=self.rng.uniform(0.2, 6)), 0.01,
                         self.rng.choice(["please stop", "leave me alone", "stop", "?"]), reverse=True)
            if hide_after is not None and i == hide_after:
                self.hides.append((ts + timedelta(minutes=30), r.recipient, r.sender))
        first = [x.ts for x in self.txs if x.rel_id == r.rel_id and x.sender == r.sender]
        r.first_abuse = min(first) if first else None

    def harassment(self):
        r = self.rel("harassment", 1)
        self._campaign(r, self.rng.uniform(0, 80), self.rng.randint(8, 60), harm=0.55, micro_p=0.7,
                       night=0.35, replies=0.25, hide_p=0.5)

    def split_threat(self):
        r = self.rel("split_threat", 1)
        self._campaign(r, self.rng.uniform(0, 80), self.rng.randint(6, 30), harm=0.35, micro_p=0.8,
                       night=0.3, replies=0.2, hide_p=0.4, split_threats=self.rng.randint(1, 3))

    def coercive_control(self):
        r = self.rel("coercive_control", 1)
        self._campaign(r, self.rng.uniform(0, 80), self.rng.randint(6, 40), harm=0.1, micro_p=0.75,
                       night=0.3, replies=0.15, hide_p=0.6)

    def abusive_support(self):
        """Legitimate payment amounts, abusive messages: the money must still arrive."""
        r, t = self.rel("abusive_child_support", 1), self.text
        amt, step, d = round(self.rng.uniform(150, 700), -1), self.rng.choice([14, 30]), self.rng.uniform(0, 14)
        while d < DAYS:
            roll = self.rng.random()
            ref = t.one(t.toxic) if roll < 0.45 else t.one(t.coercive) if roll < 0.8 else t.one(t.support)
            self.pay(r, self.when(d, 0.1), amt, ref)
            d += step
        r.first_abuse = min(x.ts for x in self.txs if x.rel_id == r.rel_id)

    def low_and_slow(self):
        r = self.rel("low_and_slow", 1)
        self._campaign(r, self.rng.uniform(0, 60), self.rng.randint(3, 6), harm=0.3, micro_p=0.4,
                       night=0.15, replies=0.1, hide_p=0.2)


INNOCENT = [("friends", 0.30), ("couple", 0.16), ("allowance", 0.08), ("rent", 0.10), ("child_support", 0.08),
            ("debt_reminder", 0.06), ("verification", 0.07), ("night_worker", 0.05), ("marketplace", 0.10)]
ABUSIVE = [("harassment", 0.35), ("split_threat", 0.2), ("coercive_control", 0.2), ("abusive_support", 0.1),
           ("low_and_slow", 0.15)]


def build(split: str, n_innocent: int, n_abusive: int, seed: int) -> World:
    b = Builder(split, seed)
    for kinds, n in ((INNOCENT, n_innocent), (ABUSIVE, n_abusive)):
        names, weights = zip(*kinds)
        for name in b.rng.choices(names, weights=weights, k=n):
            getattr(b, name)()
    b.txs.sort(key=lambda x: x.ts)
    b.hides.sort()
    return World(b.customers, b.rels, b.txs, b.hides,
                 meta={"split": split, "seed": seed, "start": START.isoformat(), "days": DAYS})
