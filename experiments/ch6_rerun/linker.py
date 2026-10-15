"""
Entity linking (Section 4.3.2): exact label match, then Sentence-BERT
(all-MiniLM-L6-v2) nearest label if its cosine similarity reaches the
threshold; otherwise the mention stays unlinked. With linking switched off
(ablation) only the exact normalised label match remains.
"""

import numpy as np

from .kb import ChainKB, norm_text


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
            return {"uri": uri, "label": kb.labels[uri], "score": 1.0, "method": "exact"}
        if not use_embeddings:
            return None
        uris = list(kb.labels)
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
