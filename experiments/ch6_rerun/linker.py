"""
Entity linking (Section 4.3.2): exact label match, then Sentence-BERT
(all-MiniLM-L6-v2) nearest label if its cosine similarity reaches the
threshold; otherwise the mention stays unlinked. With linking switched off
(ablation) only the exact normalised label match remains.
"""

import re

import numpy as np

from .kb import ChainKB, norm_text

# Sentence embeddings place opposite events close together ("demand increases" /
# "demand decreases" ~0.88, "no temperature rise" / "temperature rises" ~0.79).
# An embedding link is therefore accepted only when mention and label agree in
# negation and in direction of change; otherwise the mention stays unlinked.
_NEGATION = re.compile(
    r"\b(?:no|not|never|none|without|lack|lacks|lacking|absence|absent|opposite|false|"
    r"fails?|failed|cannot|can't|doesn't|don't|didn't|isn't|aren't|won't|non)\b|n't\b"
)
_UP = re.compile(
    r"\b(?:increas\w*|ris(?:e|es|ing|en)|rose|grow\w*|grew|elevat\w*|heighten\w*|higher|more|"
    r"strengthen\w*|gain\w*|rais\w*|boost\w*|expand\w*|accelerat\w*|add|adds|added|adding|up)\b"
)
_DOWN = re.compile(
    r"\b(?:decreas\w*|fall\w*|fell|drop\w*|declin\w*|reduc\w*|lower\w*|less|weaken\w*|"
    r"los[se]\w*|shrink\w*|shrank|diminish\w*|contract\w*|remov\w*|withdr[ae]w\w*|subtract\w*|down)\b"
)


_CONNECTIVES = {"and", "or", "if", "then", "either", "both", "least"}
_VAR_NEXT = {
    "and",
    "or",
    "is",
    "are",
    "holds",
    "hold",
    "follows",
    "implies",
    "then",
    ",",
}


def _logic_signature(t: str) -> tuple[frozenset, frozenset]:
    """Propositional variables (single letters) and connectives in a mention."""
    toks = re.findall(r"[a-z]+|,", re.sub(r"-", " ", t))
    variables = set()
    for i, tok in enumerate(toks):
        if len(tok) == 1 and tok.isalpha():
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if tok != "a" or nxt in _VAR_NEXT or not nxt:
                variables.add(tok)
    return frozenset(variables), frozenset(x for x in toks if x in _CONNECTIVES)


def polarity_signature(text: str) -> tuple:
    """What an embedding link must preserve: negation, direction of change, and
    for propositional statements the variables and connectives."""
    from .props import relation_words

    t = text.lower()
    negations = len(_NEGATION.findall(t)) % 2 == 1  # parity: "not-Q is false" means Q
    return (
        negations,
        bool(_UP.search(t)),
        bool(_DOWN.search(t)),
        *_logic_signature(t),
        relation_words(t),
    )


class Encoder:
    """Sentence-BERT encoder with a cache, shared across chains and systems."""

    def __init__(self, model_name: str, device: str = "cpu", model=None):
        self.model_name = model_name
        self.device = device
        self._model = model
        self._cache: dict[str, np.ndarray] = {}

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = np.asarray(
                self._get().encode(missing, normalize_embeddings=True), dtype=np.float32
            )
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1))


class EntityLinker:
    def __init__(self, encoder: Encoder, threshold: float = 0.7):
        self.encoder = encoder
        self.threshold = threshold

    def link(self, mention: str, kb: ChainKB, use_embeddings: bool) -> dict | None:
        key = norm_text(mention)
        if not key:
            return None
        if key in kb.by_norm:
            uri = kb.by_norm[key]
            return {
                "uri": uri,
                "label": kb.labels[uri],
                "score": 1.0,
                "method": "exact",
            }
        if not use_embeddings:
            return None
        sig = polarity_signature(key)
        # a mention much longer than a label usually merges two events
        uris = [
            u
            for u in kb.labels
            if polarity_signature(kb.labels[u]) == sig
            and len(key.split()) <= len(kb.labels[u].split()) + 2
        ]
        if not uris:
            return None
        labels = [norm_text(kb.labels[u]) for u in uris]
        mat = self.encoder.encode(labels)
        q = self.encoder.encode([key])[0]
        sims = mat @ q
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.threshold:
            return None
        return {
            "uri": uris[best],
            "label": kb.labels[uris[best]],
            "score": round(min(score, 1.0), 4),
            "method": "embedding",
        }
