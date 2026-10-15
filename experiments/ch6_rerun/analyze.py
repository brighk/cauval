"""
Tables and statistics for the Chapter 6 re-run.

    python -m experiments.ch6_rerun.analyze experiments/ch6_rerun/results/full

Metrics follow the Chapter 6 definitions and are computed on the perturbed
prompts (225 instances in the full benchmark); the original prompts are used
only as the reference for semantic invariance. Every system's final
propositions are scored by the same evaluator.

For each metric, the value per seed and the mean and standard deviation over
seeds are reported. Full CAF is compared with every other system on
per-chain entailment accuracy (averaged over prompts and seeds): mean
difference, 95% bootstrap confidence interval over chains, Wilcoxon
signed-rank test, and Holm correction within each family (baselines,
ablations).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .config import ABLATIONS, BASELINES, SYSTEMS

METRICS = [
    ("entailment", "Entailment accuracy (%)"),
    ("contradiction_flagged", "Contradiction detection (%)"),
    ("injected_recall", "Injected contradictions flagged (%)"),
    ("false_alarm", "Flags on clean chains (%)"),
    ("final_contradiction", "Final output contradicts KB (%)"),
    ("depth_ratio", "Inference depth ratio (%)"),
    ("accepted", "Accepted (%)"),
    ("si", "Semantic invariance (%)"),
    ("n_iterations", "Mean iterations"),
]
DOMAINS = ["physics", "biology", "economics", "logic", "causality"]


def load(out: Path):
    manifest = json.loads((out / "manifest.json").read_text())
    recs = defaultdict(list)
    for sdir in sorted(out.glob("seed_*")):
        seed = int(sdir.name.split("_")[1])
        for f in sorted(sdir.glob("*.jsonl")):
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            recs[(seed, f.stem)].append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return manifest, recs


def seed_metrics(rs: list[dict]) -> dict:
    pert = [r for r in rs if r["variant"] != "original"]
    if not pert:
        return {}
    f = lambda key, sub=pert: 100 * float(np.mean([r["final"][key] for r in sub])) if sub else float("nan")
    inj = [r for r in pert if r["has_injected_contradiction"]]
    clean = [r for r in pert if not r["has_injected_contradiction"]]
    m = {
        "entailment": f("entailment"),
        "contradiction_flagged": f("contradiction_flagged"),
        "injected_recall": f("contradiction_flagged", inj),
        "false_alarm": f("contradiction_flagged", clean),
        "final_contradiction": f("final_contradiction"),
        "depth_ratio": f("depth_ratio"),
        "accepted": f("accepted"),
        "n_iterations": float(np.mean([r["final"]["n_iterations"] for r in pert])),
        "n": len(pert),
    }
    orig = {r["chain_id"]: r["final"]["eval_score"] for r in rs if r["variant"] == "original"}
    per_chain = defaultdict(list)
    for r in pert:
        if r["chain_id"] in orig:
            per_chain[r["chain_id"]].append(1 - abs(orig[r["chain_id"]] - r["final"]["eval_score"]))
    m["si"] = 100 * float(np.mean([np.mean(v) for v in per_chain.values()])) if per_chain else float("nan")
    for d in DOMAINS:
        sub = [r for r in pert if r["domain"] == d]
        m[f"dom_{d}"] = f("entailment", sub)
    return m


def per_chain_entailment(recs, system, seeds) -> dict[str, float]:
    vals = defaultdict(list)
    for s in seeds:
        for r in recs.get((s, system), []):
            if r["variant"] != "original":
                vals[r["chain_id"]].append(r["final"]["entailment"])
    return {c: float(np.mean(v)) for c, v in vals.items()}


def paired(recs, a, b, seeds, rng) -> dict:
    ea, eb = per_chain_entailment(recs, a, seeds), per_chain_entailment(recs, b, seeds)
    chains = sorted(set(ea) & set(eb))
    d = np.array([ea[c] - eb[c] for c in chains]) * 100
    if len(d) == 0:
        return {}
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    per_seed = []
    for s in seeds:
        xa, xb = per_chain_entailment(recs, a, [s]), per_chain_entailment(recs, b, [s])
        cs = set(xa) & set(xb)
        per_seed.append(100 * float(np.mean([xa[c] - xb[c] for c in cs])) if cs else float("nan"))
    return {"diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": p, "n_chains": len(d), "per_seed": per_seed}


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in ps.items() if not np.isnan(p))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def fmt(vals: list[float], digits=1) -> str:
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--allow-mock", action="store_true", help="for testing the analysis code only")
    args = ap.parse_args(argv)

    manifest, recs = load(args.out)
    if manifest.get("backend") == "mock" and not args.allow_mock:
        sys.exit("These results come from the scripted mock backend, not a language model. Refusing.")
    seeds = sorted({s for s, _ in recs})
    systems = [s for s in SYSTEMS if any((sd, s) in recs for sd in seeds)]
    n_exp = manifest["n_instances"]
    warnings = [f"seed {s} / {name}: {len(recs.get((s, name), []))} of {n_exp} instances"
                for s in seeds for name in systems if len(recs.get((s, name), [])) != n_exp]

    per = {(s, name): seed_metrics(recs[(s, name)]) for s in seeds for name in systems if (s, name) in recs}
    rng = np.random.default_rng(0)
    comps = {}
    for fam in (BASELINES, ABLATIONS):
        fam_res = {b: paired(recs, "caf", b, seeds, rng) for b in fam if b in systems and "caf" in systems}
        adj = holm({b: r["p"] for b, r in fam_res.items() if r})
        for b, r in fam_res.items():
            if r:
                r["p_holm"] = adj.get(b, float("nan"))
        comps.update(fam_res)

    lines = [f"# Chapter 6 re-run: {args.out.name}", "",
             f"Model: {manifest['model']} ({manifest['backend']}). Seeds: {seeds}. "
             f"Chains: {manifest['n_chains']}. Instances per system and seed: {n_exp} "
             f"(metrics on the perturbed prompts). Values are mean ± SD over seeds.", ""]
    if manifest.get("smoke"):
        lines += ["**Smoke test only: not for reporting.**", ""]
    if warnings:
        lines += ["**Incomplete:**", *[f"- {w}" for w in warnings], ""]

    def table(names, metric_keys, title):
        lines.extend([f"## {title}", "", "| System | " + " | ".join(lbl for _, lbl in metric_keys) + " |",
                      "|---|" + "---|" * len(metric_keys)])
        for name in names:
            if name not in systems:
                continue
            cells = [fmt([per[(s, name)].get(k, float("nan")) for s in seeds if (s, name) in per],
                         2 if k == "n_iterations" else 1) for k, _ in metric_keys]
            lines.append(f"| {SYSTEMS[name].label} | " + " | ".join(cells) + " |")
        lines.append("")

    table(["caf"] + BASELINES, METRICS[:8], "Primary results (Table 6.1)")
    table(["caf"] + BASELINES, [(f"dom_{d}", d.capitalize()) for d in DOMAINS], "Per-domain entailment accuracy (Table 6.2)")
    table(["caf"] + ABLATIONS, [METRICS[0], METRICS[7], METRICS[1], METRICS[6], METRICS[8]], "Ablation (Table 6.4)")

    lines += ["## Full CAF minus each system, entailment accuracy (percentage points)", "",
              "| Compared with | Difference | 95% CI | Wilcoxon p | Holm p | Per seed |", "|---|---|---|---|---|---|"]
    for name in BASELINES + ABLATIONS:
        r = comps.get(name)
        if r:
            lines.append(f"| {SYSTEMS[name].label} | {r['diff']:+.1f} | [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | "
                         f"{r['p']:.3g} | {r.get('p_holm', float('nan')):.3g} | "
                         + ", ".join(f"{x:+.1f}" for x in r["per_seed"]) + " |")
    lines.append("")

    (args.out / "summary.md").write_text("\n".join(lines))
    (args.out / "summary.json").write_text(json.dumps(
        {"seeds": seeds, "per_seed": {f"{s}/{n}": m for (s, n), m in per.items()},
         "comparisons": comps, "warnings": warnings}, indent=2))
    (args.out / "tables.tex").write_text(latex(per, comps, seeds, systems))
    print("\n".join(lines))
    print(f"Wrote summary.md, summary.json, tables.tex to {args.out}")


def latex(per, comps, seeds, systems) -> str:
    def cell(name, k):
        vals = [per[(s, name)][k] for s in seeds if (s, name) in per and k in per[(s, name)]]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "--"
        if len(vals) == 1:
            return f"{vals[0]:.1f}"
        return f"{np.mean(vals):.1f} $\\pm$ {np.std(vals, ddof=1):.1f}"

    out = ["% Generated by experiments/ch6_rerun/analyze.py", "",
           "\\begin{tabular}{lccc}", "\\toprule",
           "System & Entailment Acc. (\\%) & Contradiction Det. (\\%) & Semantic Inv. (\\%) \\\\", "\\midrule"]
    for name in ["caf"] + BASELINES:
        if name in systems:
            out.append(f"{SYSTEMS[name].label} & {cell(name, 'entailment')} & "
                       f"{cell(name, 'contradiction_flagged')} & {cell(name, 'si')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "Configuration & Entailment Acc. (\\%) & $\\Delta$ vs Full CAF & 95\\% CI & Semantic Inv. (\\%) \\\\",
            "\\midrule"]
    for name in ["caf"] + ABLATIONS:
        if name not in systems:
            continue
        r = comps.get(name)
        z = lambda x: round(-x, 1) + 0.0  # avoid "-0.0"
        delta = f"{z(r['diff']):+.1f}" if r else "--"
        ci = f"[{z(r['ci'][1]):+.1f}, {z(r['ci'][0]):+.1f}]" if r else "--"
        out.append(f"{SYSTEMS[name].label} & {cell(name, 'entailment')} & {delta} & {ci} & {cell(name, 'si')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
