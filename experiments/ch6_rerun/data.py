"""
Benchmark instances for the Chapter 6 re-run.

The chains come from the same generator and seed as the thesis run
(SyntheticDatasetGenerator(seed=42): 75 chains, 15 per domain, 10 with an
injected contradiction, 3 perturbed prompts each = 225 instances). Each chain
also keeps its unperturbed prompt, which semantic invariance compares against.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from experiments.synthetic_dataset import CausalChain, SyntheticDatasetGenerator

from .config import DatasetSettings


@dataclass
class Instance:
    instance_id: str  # "<chain_id>::<variant>"
    chain_id: str
    domain: str
    depth: int
    variant: str  # "original" or the perturbation type
    prompt: str
    has_injected_contradiction: bool
    injected_contradiction: str | None


def load_chains(settings: DatasetSettings) -> list[CausalChain]:
    gen = SyntheticDatasetGenerator(seed=settings.seed)
    return gen.generate_dataset(
        num_chains=settings.num_chains,
        min_depth=settings.min_depth,
        max_depth=settings.max_depth,
        perturbations_per_chain=settings.perturbations_per_chain,
        contradiction_rate=settings.contradiction_rate,
        domains=list(settings.domains),
    )


def _insert_premise(prompt: str, premise: str, chain_id: str) -> str:
    """Insert '- premise' among the premise lines, at a position fixed per chain."""
    lines = prompt.split("\n")
    idx = [i for i, line in enumerate(lines) if line.startswith("- ")]
    if not idx:
        return prompt
    rng = random.Random(int(hashlib.sha256(chain_id.encode()).hexdigest()[:8], 16))
    pos = rng.randint(idx[0], idx[-1] + 1)
    return "\n".join(lines[:pos] + [f"- {premise}"] + lines[pos:])


def build_instances(chains: list[CausalChain], settings: DatasetSettings) -> list[Instance]:
    out = []
    for c in chains:
        contra = c.injected_contradictions[0] if c.injected_contradictions else None
        variants = [("original", c.to_prompt())] + [
            (p.perturbation_type.value, p.perturbed_prompt) for p in c.perturbations
        ]
        for name, prompt in variants:
            if contra and settings.contradiction_in_prompt:
                prompt = _insert_premise(prompt, contra, c.chain_id)
            out.append(
                Instance(
                    instance_id=f"{c.chain_id}::{name}",
                    chain_id=c.chain_id,
                    domain=c.domain,
                    depth=c.depth,
                    variant=name,
                    prompt=prompt,
                    has_injected_contradiction=contra is not None,
                    injected_contradiction=contra,
                )
            )
    return out


def smoke_subset(chains: list[CausalChain], instances: list[Instance]):
    """Three chains from three domains, at least one with an injected
    contradiction; the original prompt plus one perturbation each."""
    picked: list[CausalChain] = []
    with_contra = next(c for c in chains if c.injected_contradictions)
    picked.append(with_contra)
    for c in chains:
        if len(picked) == 3:
            break
        if c.domain not in {p.domain for p in picked} and 3 <= c.depth <= 6:
            picked.append(c)
    ids = {c.chain_id for c in picked}
    keep = []
    seen: dict[str, int] = {}
    for inst in instances:
        if inst.chain_id not in ids:
            continue
        n = seen.get(inst.chain_id, 0)
        if inst.variant == "original" or n < 1:
            keep.append(inst)
            if inst.variant != "original":
                seen[inst.chain_id] = n + 1
    return [c for c in chains if c.chain_id in ids], keep


def dataset_fingerprint(chains: list[CausalChain], instances: list[Instance]) -> str:
    payload = json.dumps(
        {
            "chains": [
                [c.chain_id, [(s.antecedent, s.relation.value, s.consequent) for s in c.steps]]
                for c in chains
            ],
            "instances": [asdict(i) for i in instances],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
