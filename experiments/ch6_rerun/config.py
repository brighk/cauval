"""
Settings and system definitions for the Chapter 6 re-run.

Every system is a real configuration of the same pipeline: the same language
model, the same base prompt, the same decoding settings, the same verifier.
Ablations switch one CAF component off; nothing is simulated.
"""

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class CAFSettings:
    """CAF loop settings (Chapter 4; values used in Chapter 6, Section 6.1)."""

    alpha: float = 0.5  # weight of a Partial outcome in S_CAF
    beta: float = 0.5  # penalty of a Contradiction outcome in S_CAF
    theta: float = 0.8  # acceptance threshold
    max_iterations: int = 5  # T_max
    phi_fail: float = 0.7  # Failed fraction above which the DE rejects (Sec. 4.5.3)
    num_samples: int = 5  # K, self-consistency samples per generation
    consistency_threshold: float = 0.6  # keep a proposition seen in >= this share of samples
    # Component switches used by the ablation study (Table 6.4)
    entity_linking: bool = True  # Sentence-BERT linking; off = exact label match only
    partial_matching: bool = True  # related-predicate (Partial) outcome
    structural_checks: bool = True  # DE graph checks ("SCM validation")
    iterative_feedback: bool = True  # off = a single verified pass, no refinement

    @property
    def iterations(self) -> int:
        return self.max_iterations if self.iterative_feedback else 1


@dataclass(frozen=True)
class SystemSpec:
    name: str
    label: str
    kind: str  # "single" (one generation, scored) or "caf" (verification loop)
    prompt_style: str = "plain"  # plain | cot | rag | rag_cot
    num_samples: int = 1  # for "single": >1 means self-consistency voting, no verification
    caf: CAFSettings | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.caf is not None:
            d["caf"]["iterations"] = self.caf.iterations
        return d


FULL = CAFSettings()

SYSTEMS: dict[str, SystemSpec] = {
    s.name: s
    for s in [
        # Baselines (Section 6.1)
        SystemSpec("vanilla", "Vanilla LLM", "single", "plain"),
        SystemSpec("cot", "Chain-of-Thought", "single", "cot"),
        SystemSpec("rag", "RAG", "single", "rag"),
        SystemSpec("rag_cot", "RAG + CoT", "single", "rag_cot"),
        # Stronger comparison: the same K-sample voting CAF uses, without verification
        SystemSpec(
            "vanilla_sc",
            "Vanilla + self-consistency",
            "single",
            "plain",
            num_samples=FULL.num_samples,
        ),
        # Full CAF and the five ablations of Table 6.4
        SystemSpec("caf", "Full CAF", "caf", caf=FULL),
        SystemSpec(
            "caf_no_feedback",
            "No Iterative Feedback",
            "caf",
            caf=replace(FULL, iterative_feedback=False),
        ),
        SystemSpec(
            "caf_no_self_consistency",
            "No Self-Consistency",
            "caf",
            caf=replace(FULL, num_samples=1),
        ),
        SystemSpec(
            "caf_no_scm",
            "No SCM Validation",
            "caf",
            caf=replace(FULL, structural_checks=False),
        ),
        SystemSpec(
            "caf_no_entity_linking",
            "No Entity Linking",
            "caf",
            caf=replace(FULL, entity_linking=False),
        ),
        SystemSpec(
            "caf_no_partial",
            "No Partial Matching",
            "caf",
            caf=replace(FULL, partial_matching=False),
        ),
    ]
}

BASELINES = ["vanilla", "cot", "rag", "rag_cot", "vanilla_sc"]
ABLATIONS = [
    "caf_no_feedback",
    "caf_no_self_consistency",
    "caf_no_scm",
    "caf_no_entity_linking",
    "caf_no_partial",
]


@dataclass(frozen=True)
class DecodingSettings:
    """Identical for every system (Section 6.1)."""

    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    max_new_tokens: int = 512


@dataclass(frozen=True)
class DatasetSettings:
    """The Chapter 6 benchmark: SyntheticDatasetGenerator(seed=42), as in the thesis run."""

    seed: int = 42
    num_chains: int = 75
    min_depth: int = 2
    max_depth: int = 10
    perturbations_per_chain: int = 3
    contradiction_rate: float = 0.2
    domains: tuple = ("physics", "biology", "economics", "logic", "causality")
    # Put each injected contradiction into the prompt as an extra premise, so the
    # "chains with injected contradictions" actually present one to the model.
    contradiction_in_prompt: bool = True


@dataclass(frozen=True)
class EvalSettings:
    entity_threshold: float = 0.7  # Sentence-BERT cosine similarity for linking
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 3  # same as the original RAG baseline
    # Every system's final propositions are re-scored by one common evaluator
    # (full linking + partial matching), so an ablation changes the system,
    # never the yardstick.
    evaluator: CAFSettings = field(default_factory=CAFSettings)
