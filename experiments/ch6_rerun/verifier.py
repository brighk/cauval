"""
Formal Verification Layer (Section 4.3, Algorithm 4.1): each proposition is
linked to the chain's knowledge base and classified with SPARQL queries as
Verified, Contradiction, Partial or Failed.
"""

from .config import CAFSettings
from .kb import ChainKB, polarity
from .linker import EntityLinker
from .props import Proposition

VERIFIED, PARTIAL, CONTRADICTION, FAILED = "VERIFIED", "PARTIAL", "CONTRADICTION", "FAILED"


class Verifier:
    def __init__(self, linker: EntityLinker):
        self.linker = linker

    def verify(self, props: list[Proposition], kb: ChainKB, settings: CAFSettings) -> list[dict]:
        return [self._one(p, kb, settings) for p in props]

    def _one(self, p: Proposition, kb: ChainKB, st: CAFSettings) -> dict:
        s = self.linker.link(p.subject, kb, st.entity_linking)
        o = self.linker.link(p.obj, kb, st.entity_linking)
        out = {
            "prop": p.text(),
            **p.to_dict(),
            "s_link": s,
            "o_link": o,
            "kb_relations": [],
            "status": FAILED,
            "reason": "",
        }
        if s is None or o is None:
            out["reason"] = "unlinked " + " and ".join(
                n for n, x in (("subject", s), ("object", o)) if x is None
            )
            return out
        if s["uri"] == o["uri"]:
            out["reason"] = "subject and object link to the same entity"
            return out

        su, ou = s["uri"], o["uri"]
        if p.relation == "independent_of":
            preds = kb.predicates(su, ou) | kb.predicates(ou, su)
            out["kb_relations"] = sorted(preds)
            if preds:
                out["status"], out["reason"] = CONTRADICTION, "knowledge base links the two"
            else:
                out["reason"] = "no support"
            return out

        preds = kb.predicates(su, ou)
        out["kb_relations"] = sorted(preds)
        sign = polarity(p.relation)
        if p.negated:
            if any(polarity(q) == sign for q in preds):
                out["status"], out["reason"] = CONTRADICTION, "knowledge base asserts the relation"
            else:
                out["reason"] = "negated claim without support"
            return out
        if kb.ask(su, p.relation, ou):
            out["status"], out["reason"] = VERIFIED, "exact triple"
        elif any(polarity(q) != sign for q in preds):
            out["status"], out["reason"] = CONTRADICTION, "opposing relation"
        elif st.partial_matching and preds:
            out["status"], out["reason"] = PARTIAL, "related predicate"
        else:
            out["reason"] = "no support"
        return out
