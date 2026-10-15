# Chapter 6 re-run

This folder re-runs the Chapter 6 evaluation with the real verification
pipeline, repeated over seeds. It answers the examiners' comment:

> Strengthen the experimental evaluation. Include stronger comparisons, further
> ablation studies and repeated runs to establish which components improve
> performance and whether the improvements are reliable.

Every number comes from Qwen2.5-7B-Instruct output that has been parsed, linked
and checked with SPARQL against a knowledge base. Nothing is simulated.

## What runs

| System | What it is |
|---|---|
| `vanilla`, `cot`, `rag`, `rag_cot` | The four Chapter 6 baselines: one generation, scored by the verifier |
| `vanilla_sc` | Vanilla prompt with the same 5-sample voting CAF uses, but no verification. This is the stronger comparison: it separates sampling from verification |
| `caf` | Full CAF (Algorithm 4.3) with α = 0.5, β = 0.5, θ = 0.8, T_max = 5 and K = 5 samples |
| `caf_no_feedback` | A single verified pass with no refinement |
| `caf_no_self_consistency` | One sample per generation instead of five |
| `caf_no_scm` | The Deterministic Executive's structural checks switched off |
| `caf_no_entity_linking` | Exact label match only, without Sentence-BERT |
| `caf_no_partial` | No Partial outcome; related predicates count as Failed |

The benchmark is the thesis one, produced by the same generator with seed 42.

| Item | Count |
|---|---|
| Chains | 75 (15 per domain) |
| Chains with an injected contradiction | 10 |
| Perturbed prompts (metrics are computed on these) | 225 |
| Original prompts (used only for semantic invariance) | 75 |

Seeds 1, 2 and 3 give three repeated runs of every system.

**Design points**
- **Knowledge base.** Each chain gets its own in-memory RDF graph. It holds every stated step and, for every pair joined by a path, the entailed `leads_to` or `prevents` relation. It is correct by construction, as Section 4.6 assumes.
- **Prompts.** All systems share one system message. It asks for `[PROP] cause | relation | effect` lines at the end of the answer. Baselines add only their original wrappers from `experiments/baselines.py`, and CAF adds only its constraints.
- **RAG retrieval.** The RAG baselines retrieve the top 3 statements of that same knowledge base by Sentence-BERT similarity, so they see the knowledge CAF checks against.
- **Injected contradictions.** Each injected contradiction is shown to the model as an extra premise. Without that, the "chains with injected contradictions" would never present one.
- **One evaluator.** Every system's final propositions are re-scored by the same evaluator, with full linking and partial matching. An ablation therefore changes the system, never the yardstick.
- **Common random numbers.** A request's seed depends on run seed, instance and iteration, not on the system. Full CAF and its ablations start from the same first draft, so paired differences come from the switched-off component.
- **Batch-invariant decoding.** `llm.py` sets `VLLM_BATCH_INVARIANT=1`. Without it, the same prompt and seed could give different text depending on the rest of the batch. In a 40-instance pilot that changed 7 instances between two identical runs; with it, none changed. It costs about 2.5 times the generation time.
- **Constraints.** CAF's refinement constraints are prepended to the next user prompt, as Section 4.5.3 states. In the system message, Qwen2.5-7B ignored them.
- **Conservative linking.** Sentence-BERT scores opposite events as near-identical: "demand decreases" against "demand increases" is 0.88, and "P and Q hold" against "P or Q holds" is 0.93. An embedding link is therefore accepted only if the mention and the label agree in negation (by parity, so "not-Q is false" means Q), direction of change, propositional variables, connectives and relation words. Otherwise the mention stays unlinked, which Section 4.3.2 calls "treated as unknown". The double-negation perturbation ("it is not the case that X does not occur") is normalised to X before linking.

## Model

Use **Qwen2.5-7B-Instruct**, the model Chapter 6 names. In a 10-chain pilot, Qwen3-14B-AWQ was at the ceiling. Its vanilla accuracy was 95%, CAF matched it, and it never repeated an injected contradiction, so neither the baselines nor the ablations could be told apart. Qwen2.5-7B left room: vanilla was 80% and CAF 88% in the same pilot.

## On the RTX 3090 PC

The repository is `~/causal-reasoning` and the environment is `/projects/ai-venv`, which has everything needed. It was tested with vLLM 0.21.0 and torch 2.11. A fresh environment can use `requirements.txt` instead.

```bash
cd ~/causal-reasoning && git status          # commit first: the manifest records the commit
PY=/projects/ai-venv/bin/python
$PY -m pytest tests/unit/test_ch6_rerun.py -q   # 21 tests, under a minute
```

**1. Smoke test.** It uses 3 chains, the original prompt plus one perturbation, seed 1 and every system. It takes about 3 minutes.

```bash
$PY -m experiments.ch6_rerun.run --smoke
```

It ends with a report table and one worked example. Check these before the full run:
- `parsed` is near 100%. This means the model writes `[PROP]` lines.
- `linked` is well above 60%. This means the propositions reach the knowledge base.
- `caf` shows a mix of decisions and, on some instances, more than one iteration.
- The example's verification lines look right for its prompt.

The report lists anything that falls short under PROBLEMS.

**2. Full run.** It uses 3 seeds, all 11 systems and 300 prompts per system and seed. Expect about 45 minutes per seed, so 2 to 3 hours in total. Run it in `tmux` so it survives a closed terminal.

```bash
tmux new -s ch6
$PY -m experiments.ch6_rerun.run --out experiments/ch6_rerun/results/full
```

If it stops, run the same command again. Finished instances are skipped. To run seeds one at a time, add `--seeds 1`, then `--seeds 2`, then `--seeds 3`, keeping the same `--out`.

**3. Analysis.**

```bash
$PY -m experiments.ch6_rerun.analyze experiments/ch6_rerun/results/full
```

This writes `summary.md`, `summary.json` and `tables.tex` next to the results. The output contains:
- mean ± SD over seeds for every Chapter 6 metric
- per-domain accuracy
- the ablation table
- for Full CAF against every other system: the paired difference in entailment accuracy, its 95% bootstrap interval over chains, a Wilcoxon signed-rank p-value with Holm correction, and the difference in each seed

The `results/` folder is git-ignored. Zip `results/full` and copy it back.

## Without vLLM

`--backend hf` uses plain transformers. It is several times slower but works on any CUDA machine. `--backend mock` is a scripted stand-in for the tests. Its runs are written to `*_MOCK` folders, and `analyze.py` refuses them.

## What the logs contain

Each line of `results/<run>/seed_<s>/<system>.jsonl` is one instance:
- per iteration: the exact system and user prompt, the request seed, all K raw samples, the propositions parsed from each sample, the votes, and the propositions kept
- per proposition: entity links with their method and score, the knowledge-base relations found, and the outcome with its reason
- per iteration: the score, the structural violations, the decision, and the constraints passed to the next iteration
- the final output with the evaluator's scoring

`manifest.json` records the model, decoding settings, dataset fingerprint, git commit, package versions and GPU.

## Files

| File | Contents |
|---|---|
| `config.py` | Settings and the 11 systems |
| `data.py` | Benchmark instances |
| `kb.py` | Per-chain RDF knowledge base and SPARQL queries |
| `linker.py` | Exact and Sentence-BERT entity linking |
| `props.py` | Proposition parsing and self-consistency voting |
| `verifier.py` | Verified / Partial / Contradiction / Failed (Algorithm 4.1) |
| `executive.py` | Score, structural checks, adjudication and constraints (Algorithm 4.2) |
| `prompts.py` | Shared system message, baseline wrappers and retrieval |
| `llm.py` | vLLM, transformers and mock backends |
| `systems.py` | Lockstep runners for single-pass systems and the CAF loop (Algorithm 4.3) |
| `run.py` | Command-line entry point, resume and smoke report |
| `analyze.py` | Tables and statistics |
