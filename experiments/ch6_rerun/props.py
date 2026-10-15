"""
Proposition extraction and normalisation (Section 4.2.3).

Structured extraction reads the '[PROP] cause | relation | effect' lines every
system is asked to end with (the unpiped form '[PROP] cause relation effect'
is accepted too). If a response has no [PROP] line, a conservative
pattern-based fallback reads causal sentences from the free text. Relations
are mapped to the causal ontology; unknown relations are dropped rather than
guessed.
"""

import re
from collections import Counter
from dataclasses import asdict, dataclass

from .kb import norm_text

# phrase -> canonical relation (negation is detected separately). Tier 1 are the
# ontology's own relation words; tier 2 are paraphrases (including the ones the
# benchmark's perturbations use). Words that also occur inside the benchmark's
# entity names ("demand increases", "current decreases") are deliberately absent.
_TIER1 = {
    "causes": "causes", "cause": "causes", "caused": "causes",
    "leads to": "leads_to", "lead to": "leads_to", "leads_to": "leads_to", "led to": "leads_to",
    "ultimately leads to": "leads_to", "eventually leads to": "leads_to",
    "indirectly leads to": "leads_to", "results in": "leads_to", "result in": "leads_to",
    "enables": "enables", "enable": "enables",
    "implies": "implies", "implies that": "implies", "imply": "implies",
    "requires": "requires", "require": "requires",
    "influences": "influences", "influence": "influences",
    "prevents": "prevents", "prevent": "prevents",
}
_TIER2 = {
    "triggers": "causes", "produces": "causes", "induces": "causes", "brings about": "causes",
    "gives rise to": "leads_to", "contributes to": "leads_to", "culminates in": "leads_to",
    "ends with": "leads_to", "resulting in": "leads_to",
    "allows": "enables", "permits": "enables", "facilitates": "enables", "makes possible": "enables",
    "entails": "implies", "means that": "implies", "suggests that": "implies", "indicates that": "implies",
    "needs": "requires", "depends on": "requires", "demands": "requires", "necessitates": "requires",
    "affects": "influences",
    "inhibits": "prevents", "blocks": "prevents", "stops": "prevents", "suppresses": "prevents",
    "counteracts": "prevents",
}
_LEXICON = {**_TIER2, **_TIER1}
# Passive forms: "B is caused by A" is read as (A, causes, B).
_PASSIVE = {
    "caused by": "causes", "triggered by": "causes", "produced by": "causes",
    "induced by": "causes", "due to": "causes", "results from": "leads_to",
    "result from": "leads_to", "resulting from": "leads_to", "the result of": "leads_to",
    "a result of": "leads_to", "a consequence of": "leads_to", "led to by": "leads_to",
    "enabled by": "enables", "made possible by": "enables", "implied by": "implies",
    "follows from": "implies", "required by": "requires", "influenced by": "influences",
    "affected by": "influences", "prevented by": "prevents", "inhibited by": "prevents",
    "blocked by": "prevents", "stopped by": "prevents",
}
_AUX = r"(?:(?:is|are|was|were|be|being|been|gets|get)\s+)"
_NEG = r"(?:does not|doesn't|do not|don't|did not|cannot|can't|will not|won't|is not|never|no longer|not)"
_PHRASES = sorted(list(_LEXICON) + list(_PASSIVE), key=len, reverse=True)
_REL_RE = re.compile(
    r"\b(?P<neg>" + _NEG + r"\s+)?" + _AUX + r"?(?P<rel>"
    + "|".join(re.escape(p) for p in _PHRASES) + r")\b",
    re.IGNORECASE,
)
_BASE_VERB = {
    "causes": "cause", "leads_to": "lead to", "enables": "enable", "implies": "imply",
    "requires": "require", "influences": "influence", "prevents": "prevent",
}
_PROP_LINE = re.compile(r"^\s*[-*\d.)\s]*\**\[PROP\]\**\s*:?\s*(?P<body>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Proposition:
    subject: str
    relation: str  # canonical relation, or "independent_of"
    obj: str
    negated: bool = False
    source: str = "prop"  # "prop" (structured) or "fallback"

    def key(self) -> tuple:
        return (norm_text(self.subject), self.relation, self.negated, norm_text(self.obj))

    def text(self) -> str:
        if self.relation == "independent_of":
            return f"{self.obj} is independent of {self.subject}"
        if self.negated:
            return f"{self.subject} does not {_BASE_VERB[self.relation]} {self.obj}"
        return f"{self.subject} {self.relation.replace('_', ' ')} {self.obj}"

    def to_dict(self) -> dict:
        return asdict(self)


_PREAMBLE = re.compile(
    r"^\s*(?:(?:we|one) (?:can|could|may) (?:conclude|infer|say|deduce) that|in conclusion,?|"
    r"this (?:means|shows|implies|suggests) that|it follows that|(?:therefore|thus|hence|so|"
    r"consequently|finally|ultimately|overall),?|if|when|given(?: that)?:?)\s+",
    re.IGNORECASE,
)
_MODAL_TAIL = re.compile(r"\s+(?:can|could|will|would|may|might|must|should|also|then|thereby)$", re.IGNORECASE)


def _clean(s: str) -> str:
    s = s.strip()
    for _ in range(3):
        s2 = _PREAMBLE.sub("", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = s.strip(" .,;:-*\"'")
    s = _MODAL_TAIL.sub("", s)
    return s.strip(" .,;:-*\"'")


def _map_relation(phrase: str) -> tuple[str | None, bool, bool]:
    """Returns (canonical relation, negated, passive)."""
    p = phrase.lower().strip(" .,:;*")
    if re.search(r"\bindependent of\b", p):
        return "independent_of", False, False
    neg = False
    m = re.match(_NEG + r"\s+(.*)$", p)
    if m:
        neg, p = True, m.group(1)
    p = re.sub(r"^" + _AUX, "", p)
    p = re.sub(r"^(ultimately|eventually|indirectly|directly|also|thereby|always)\s+", "", p)
    if p in _PASSIVE:
        return _PASSIVE[p], neg, True
    if p in _LEXICON:
        return _LEXICON[p], neg, False
    # verb inflections after a negation ("does not cause")
    for cand in (p + "s", p.rstrip("s")):
        if cand in _LEXICON:
            return _LEXICON[cand], neg, False
    return None, neg, False


def _split_free(body: str, source: str) -> Proposition | None:
    body = body.strip()
    m = re.match(r"^the opposite of (.+?) occurs when (.+)$", body, re.IGNORECASE)
    if m:
        return Proposition(_clean(m.group(2)), "prevents", _clean(m.group(1)), False, source)
    m = re.match(r"^(.+?) is independent of (.+)$", body, re.IGNORECASE)
    if m:
        return Proposition(_clean(m.group(2)), "independent_of", _clean(m.group(1)), False, source)
    m = re.match(r"^if (.+?),\s*then\s+(?:ultimately\s+|eventually\s+)?(.+)$", body, re.IGNORECASE)
    if m:
        return Proposition(_clean(m.group(1)), "leads_to", _clean(m.group(2)), False, source)
    # Several relation words may match; prefer tier 1, then the leftmost.
    matches = [m for m in _REL_RE.finditer(body) if 0 < m.start() and m.end() < len(body)]
    if not matches:
        return None
    tier = lambda m: 0 if (m.group("rel").lower() in _TIER1 or m.group("rel").lower() in _PASSIVE) else 1
    m = min(matches, key=lambda m: (tier(m), m.start()))
    subj, obj = _clean(body[: m.start()]), _clean(body[m.end() :])
    word = m.group("rel").lower()
    if word in _PASSIVE:
        rel, (subj, obj) = _PASSIVE[word], (obj, subj)
    else:
        rel = _LEXICON[word]
    if not subj or not obj or len(subj.split()) > 8 or len(obj.split()) > 8:
        return None
    return Proposition(subj, rel, obj, bool(m.group("neg")), source)


def parse_response(text: str) -> list[Proposition]:
    props: list[Proposition] = []
    for line in text.splitlines():
        m = _PROP_LINE.match(line)
        if not m:
            continue
        body = m.group("body").strip().strip("*").strip()
        parts = [p.strip() for p in body.split("|")]
        p = None
        if len(parts) == 3 and all(parts):
            rel, neg, passive = _map_relation(parts[1])
            if rel == "independent_of" or passive:
                p = Proposition(_clean(parts[2]), rel, _clean(parts[0]), neg, "prop")
            elif rel:
                p = Proposition(_clean(parts[0]), rel, _clean(parts[2]), neg, "prop")
        elif "|" not in body:
            p = _split_free(body, "prop")
        if p and p.subject and p.obj:
            props.append(p)
    if props:
        return _dedupe(props)
    # Fallback: causal sentences in the free text.
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        sent = re.sub(r"^\s*[-*\d.)]+\s*", "", sent).strip()
        if not sent or len(sent.split()) > 25 or "[PROP]" in sent.upper() or "|" in sent:
            continue
        p = _split_free(sent.rstrip("."), "fallback")
        if p:
            props.append(p)
    return _dedupe(props)


def _dedupe(props: list[Proposition]) -> list[Proposition]:
    seen, out = set(), []
    for p in props:
        if p.key() not in seen:
            seen.add(p.key())
            out.append(p)
    return out


def consensus(samples: list[list[Proposition]], threshold: float):
    """Self-consistency (Section 4.2.2): keep propositions recurring in at least
    `threshold` of the samples (each sample votes once per proposition).
    Returns (kept propositions, index of the most representative sample, votes)."""
    k = len(samples)
    votes: Counter = Counter()
    first: dict[tuple, Proposition] = {}
    keysets = []
    for props in samples:
        keys = {p.key() for p in props}
        for p in props:
            first.setdefault(p.key(), p)
        votes.update(keys)
        keysets.append(keys)
    kept = {key for key, n in votes.items() if n / k >= threshold}

    def agreement(keys):
        union = keys | kept
        return len(keys & kept) / len(union) if union else 1.0

    best = max(range(k), key=lambda i: agreement(keysets[i])) if k else 0
    kept_props = [p for key, p in first.items() if key in kept]
    vote_list = [
        {"prop": first[key].text(), "votes": n} for key, n in votes.most_common()
    ]
    return kept_props, best, vote_list
