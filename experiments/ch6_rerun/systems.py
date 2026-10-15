"""
Runners. All instances of one system advance together, one generation round
at a time, so each round is a single batched call to the backend.

Seeds use common random numbers: a request's seed depends on the run seed,
the instance and the iteration, not on the system. Full CAF and its ablations
therefore start from the same first draft, and paired differences reflect the
component that was switched off.
"""

import time
from dataclasses import asdict
from typing import Callable

from .config import CAFSettings, EvalSettings, SystemSpec
from .data import Instance
from .executive import (
    ACCEPT,
    REFINE,
    adjudicate,
    caf_score,
    entailment,
    structural_violations,
)
from .kb import ChainKB
from .linker import Encoder
from .llm import GenRequest, request_seed
from .prompts import retrieve, system_message, user_message, with_constraints
from .props import Proposition, consensus, parse_response
from .verifier import CONTRADICTION, Verifier


class Context:
    def __init__(
        self,
        kbs: dict[str, ChainKB],
        verifier: Verifier,
        encoder: Encoder,
        backend,
        ev: EvalSettings,
    ):
        self.kbs = kbs
        self.verifier = verifier
        self.encoder = encoder
        self.backend = backend
        self.ev = ev


def _depth(inst: Instance, logs: list[dict], accepted: bool) -> dict:
    """Inference depth as defined in experiments/metrics.py (thesis Table 6.1)."""
    max_depth, cdepth = inst.depth, None
    for log in logs:
        for i, r in enumerate(log["results"]):
            if r["status"] == CONTRADICTION:
                cdepth = i + 1
                max_depth = min(max_depth, i)
                break
    if accepted and cdepth is None:
        max_depth = inst.depth
    return {
        "depth_achieved": max_depth,
        "contradiction_at": cdepth,
        "depth_ratio": max_depth / inst.depth if inst.depth else 0.0,
    }


def _finalise(
    spec: SystemSpec,
    inst: Instance,
    logs: list[dict],
    ctx: Context,
    run_seed: int,
    decision: str | None,
) -> dict:
    last = logs[-1]
    kb = ctx.kbs[inst.chain_id]
    ev = ctx.ev.evaluator

    props = [Proposition(**p) for p in last["props"]]
    eval_results = ctx.verifier.verify(props, kb, ev)
    eval_score = caf_score(eval_results, ev)
    eval_viol = structural_violations(eval_results)
    accepted = (
        decision == ACCEPT
        if decision
        else (bool(eval_results) and eval_score >= ev.theta and not eval_viol)
    )
    return {
        "run_seed": run_seed,
        "system": spec.name,
        **asdict(inst),
        "iterations": logs,
        "final": {
            "text": last["samples"][last["representative"]],
            "props": last["props"],
            "decision": decision,
            "n_iterations": len(logs),
            "system_score": last["score"],
            "eval_results": eval_results,
            "eval_score": eval_score,
            "eval_violations": eval_viol,
            "entailment": entailment(eval_results),
            "contradiction_flagged": any(
                r["status"] == CONTRADICTION for log in logs for r in log["results"]
            ),
            "final_contradiction": any(
                r["status"] == CONTRADICTION for r in eval_results
            ),
            "accepted": accepted,
            **_depth(inst, logs, accepted),
        },
    }


def _round_log(
    t,
    req: GenRequest,
    samples,
    sample_props,
    props,
    rep,
    votes,
    results,
    score,
    violations,
    decision,
    constraints,
) -> dict:
    return {
        "t": t,
        "system_prompt": req.system,
        "user_prompt": req.user,
        "seed": req.seed,
        "samples": samples,
        "sample_props": [[p.to_dict() for p in ps] for ps in sample_props],
        "props": [p.to_dict() for p in props],
        "representative": rep,
        "votes": votes,
        "results": results,
        "score": score,
        "violations": violations,
        "decision": decision,
        "constraints_next": constraints,
    }


def _select(sample_props, k: int, threshold: float):
    if k > 1:
        return consensus(sample_props, threshold)
    return sample_props[0], 0, None


def run_single(
    spec: SystemSpec,
    insts: list[Instance],
    ctx: Context,
    run_seed: int,
    write: Callable[[dict], None],
) -> None:
    """One generation per instance (K samples with voting for vanilla_sc), scored
    by the evaluator. No verification feeds back into generation."""
    ev = ctx.ev.evaluator
    reqs = []
    for inst in insts:
        facts = None
        if spec.prompt_style in ("rag", "rag_cot"):
            facts = retrieve(
                inst.prompt, ctx.kbs[inst.chain_id], ctx.encoder, ctx.ev.rag_top_k
            )
        reqs.append(
            GenRequest(
                system_message(),
                user_message(spec.prompt_style, inst.prompt, facts),
                spec.num_samples,
                request_seed(run_seed, inst.instance_id, 0),
            )
        )
    outs = ctx.backend.generate(reqs)
    for inst, req, samples in zip(insts, reqs, outs):
        sample_props = [parse_response(x) for x in samples]
        props, rep, votes = _select(
            sample_props, spec.num_samples, ev.consistency_threshold
        )
        results = ctx.verifier.verify(props, ctx.kbs[inst.chain_id], ev)
        log = _round_log(
            0,
            req,
            samples,
            sample_props,
            props,
            rep,
            votes,
            results,
            caf_score(results, ev),
            [],
            None,
            [],
        )
        write(_finalise(spec, inst, [log], ctx, run_seed, None))


def run_caf(
    spec: SystemSpec,
    insts: list[Instance],
    ctx: Context,
    run_seed: int,
    write: Callable[[dict], None],
) -> None:
    """Algorithm 4.3, run for all instances in lockstep."""
    st: CAFSettings = spec.caf
    states = [{"inst": i, "constraints": [], "logs": []} for i in insts]
    active = states
    for t in range(st.iterations):
        if not active:
            break
        t0 = time.time()
        reqs = [
            GenRequest(
                system_message(),
                with_constraints(
                    user_message("plain", s["inst"].prompt, None), s["constraints"]
                ),
                st.num_samples,
                request_seed(run_seed, s["inst"].instance_id, t),
            )
            for s in active
        ]
        outs = ctx.backend.generate(reqs)
        nxt = []
        for s, req, samples in zip(active, reqs, outs):
            inst = s["inst"]
            sample_props = [parse_response(x) for x in samples]
            props, rep, votes = _select(
                sample_props, st.num_samples, st.consistency_threshold
            )
            results = ctx.verifier.verify(props, ctx.kbs[inst.chain_id], st)
            violations = structural_violations(results) if st.structural_checks else []
            adj = adjudicate(results, violations, t, st, any(sample_props))
            s["logs"].append(
                _round_log(
                    t,
                    req,
                    samples,
                    sample_props,
                    props,
                    rep,
                    votes,
                    results,
                    adj["score"],
                    violations,
                    adj["decision"],
                    adj["constraints"],
                )
            )
            if adj["decision"] == REFINE:
                s["constraints"] = adj["constraints"]
                nxt.append(s)
            else:
                write(_finalise(spec, inst, s["logs"], ctx, run_seed, adj["decision"]))
        print(
            f"    {spec.name} round {t + 1}: {len(active)} generated, {len(nxt)} refining "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        active = nxt


def run_system(spec: SystemSpec, insts, ctx, run_seed, write) -> None:
    if spec.kind == "caf":
        run_caf(spec, insts, ctx, run_seed, write)
    else:
        run_single(spec, insts, ctx, run_seed, write)
