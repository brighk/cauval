"""
Chapter 6 re-run: CAF, four baselines, a self-consistency baseline and the five
ablations, all with the real verifier, over several seeds.

Smoke test (3 chains x 2 prompts, seed 1, every system; ~10-20 min on a 3090):
    python -m experiments.ch6_rerun.run --smoke

Full run (75 chains x 4 prompts, seeds 1-3, every system):
    python -m experiments.ch6_rerun.run --out experiments/ch6_rerun/results/full

Interrupted runs resume: finished instances are read back from the JSONL files
and skipped. Then: python -m experiments.ch6_rerun.analyze <out dir>
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import SYSTEMS, DatasetSettings, DecodingSettings, EvalSettings
from .data import build_instances, dataset_fingerprint, load_chains, smoke_subset
from .kb import build_kb
from .linker import Encoder, EntityLinker
from .llm import VLLM_ENV, make_backend
from .systems import Context, run_system
from .verifier import Verifier

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return ""


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in (
        "torch",
        "transformers",
        "vllm",
        "sentence_transformers",
        "rdflib",
        "networkx",
    ):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["instance_id"])
                except json.JSONDecodeError:
                    pass  # a line cut off by an interruption; that instance reruns
    return ids


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="3 chains, original + 1 perturbation, seed 1",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS)
    )
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--chains", type=int, default=None, help="use only the first N chains"
    )
    ap.add_argument(
        "--chunk", type=int, default=150, help="instances per lockstep block"
    )
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--hf-batch-size", type=int, default=8)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--no-contradiction-in-prompt", action="store_true")
    args = ap.parse_args(argv)

    ds = DatasetSettings(contradiction_in_prompt=not args.no_contradiction_in_prompt)
    dec = DecodingSettings()
    ev = EvalSettings()

    chains = load_chains(ds)
    instances = build_instances(chains, ds)
    if args.smoke:
        chains, instances = smoke_subset(chains, instances)
        args.seeds = args.seeds if args.seeds != [1, 2, 3] else [1]
    elif args.chains:
        chains = chains[: args.chains]
        keep = {c.chain_id for c in chains}
        instances = [i for i in instances if i.chain_id in keep]

    if args.out is None:
        tag = "smoke" if args.smoke else "run"
        args.out = HERE / "results" / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.backend == "mock":
        args.out = args.out.with_name(args.out.name + "_MOCK")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "backend": args.backend,
        "model": args.model
        if args.backend != "mock"
        else "MOCK (scripted, not a language model)",
        "dataset": asdict(ds),
        "dataset_fingerprint": dataset_fingerprint(chains, instances),
        "n_chains": len(chains),
        "n_instances": len(instances),
        "decoding": asdict(dec),
        "evaluation": asdict(ev),
        "systems": {name: SYSTEMS[name].to_dict() for name in args.systems},
        "smoke": args.smoke,
    }
    mpath = args.out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        for key in (
            "backend",
            "model",
            "dataset_fingerprint",
            "decoding",
            "evaluation",
        ):
            if old.get(key) != manifest[key]:
                sys.exit(
                    f"{mpath} was written with a different {key}; use a new --out directory."
                )
        manifest["systems"] = {**old.get("systems", {}), **manifest["systems"]}
        manifest["seeds"] = sorted(set(old.get("seeds", [])) | set(args.seeds))
        manifest["sessions"] = old.get("sessions", [])
    else:
        manifest["seeds"] = sorted(args.seeds)
        manifest["sessions"] = []
    manifest["sessions"].append(
        {
            "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(
                _git("status", "--porcelain", "--", "experiments/ch6_rerun")
            ),
            "argv": sys.argv[1:] if argv is None else argv,
            "versions": _versions(),
            "env": {k: os.environ.get(k) for k in VLLM_ENV},
        }
    )
    mpath.write_text(json.dumps(manifest, indent=2))

    print(f"Output: {args.out}")
    print(
        f"{len(chains)} chains, {len(instances)} instances, seeds {args.seeds}, "
        f"systems {', '.join(args.systems)}"
    )

    kbs = {c.chain_id: build_kb(c) for c in chains}
    encoder = Encoder(ev.embedding_model, device=args.embed_device)
    verifier = Verifier(EntityLinker(encoder, ev.entity_threshold))
    backend = make_backend(
        args.backend,
        args.model,
        dec,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.hf_batch_size,
    )
    ctx = Context(kbs, verifier, encoder, backend, ev)

    timings = {}
    for seed in args.seeds:
        sdir = args.out / f"seed_{seed}"
        sdir.mkdir(exist_ok=True)
        for name in args.systems:
            spec = SYSTEMS[name]
            path = sdir / f"{name}.jsonl"
            done = _done_ids(path)
            todo = [i for i in instances if i.instance_id not in done]
            if not todo:
                print(f"[seed {seed}] {name}: complete ({len(done)})")
                continue
            print(
                f"[seed {seed}] {name}: {len(todo)} to run ({len(done)} done)",
                flush=True,
            )
            t0 = time.time()
            with path.open("a") as f:

                def write(rec, f=f):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()

                for start in range(0, len(todo), args.chunk):
                    run_system(spec, todo[start : start + args.chunk], ctx, seed, write)
            timings[f"seed_{seed}/{name}"] = round(time.time() - t0, 1)
            print(
                f"[seed {seed}] {name}: done in {timings[f'seed_{seed}/{name}']:.0f}s",
                flush=True,
            )

    manifest = json.loads(mpath.read_text())
    manifest["sessions"][-1]["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["sessions"][-1]["timings_s"] = timings
    mpath.write_text(json.dumps(manifest, indent=2))

    if args.smoke:
        smoke_report(args.out, args.seeds[0], args.systems)


def smoke_report(out: Path, seed: int, systems: list[str]) -> None:
    """Checks that the pipeline produced parseable, linkable output for every system."""
    print("\n=== Smoke test report ===")
    print(
        f"{'system':26s} {'inst':>4s} {'parsed':>7s} {'linked':>7s} {'ent.acc':>7s} "
        f"{'iters':>5s}  statuses / decisions"
    )
    problems = []
    for name in systems:
        recs = [
            json.loads(line) for line in (out / f"seed_{seed}" / f"{name}.jsonl").open()
        ]
        samples = [
            s for r in recs for log in r["iterations"] for s in log["sample_props"]
        ]
        parsed = sum(1 for s in samples if s) / max(len(samples), 1)
        res = [x for r in recs for x in r["final"]["eval_results"]]
        linked = sum(1 for x in res if x["s_link"] and x["o_link"]) / max(len(res), 1)
        acc = sum(r["final"]["entailment"] for r in recs) / max(len(recs), 1)
        iters = sum(r["final"]["n_iterations"] for r in recs) / max(len(recs), 1)
        st = Counter(x["status"] for x in res)
        dec = Counter(r["final"]["decision"] for r in recs if r["final"]["decision"])
        print(
            f"{name:26s} {len(recs):4d} {parsed:7.0%} {linked:7.0%} {acc:7.3f} {iters:5.1f}  "
            f"{dict(st)} {dict(dec) if dec else ''}"
        )
        if parsed < 0.8:
            problems.append(
                f"{name}: only {parsed:.0%} of samples yielded a proposition"
            )
        if res and linked < 0.6:
            problems.append(
                f"{name}: only {linked:.0%} of final propositions linked to the KB"
            )
    ex = (
        [json.loads(line) for line in (out / f"seed_{seed}" / "caf.jsonl").open()]
        if "caf" in systems
        else []
    )
    if ex:
        r = ex[0]
        print(
            f"\nExample (caf, {r['instance_id']}):\n--- prompt ---\n{r['prompt']}\n--- final answer ---\n"
            f"{r['final']['text'][:1500]}\n--- verification ---"
        )
        for x in r["final"]["eval_results"]:
            print(f"  {x['status']:13s} {x['prop']}  ({x['reason']})")
    print(
        "\nPROBLEMS:\n  " + "\n  ".join(problems)
        if problems
        else "\nNo problems detected."
    )
    print(f"Logs: {out}")


if __name__ == "__main__":
    main()
