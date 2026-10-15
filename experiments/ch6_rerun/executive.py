"""
Deterministic Executive (Section 4.5): CAF score, structural consistency
checks on the proposition graph, adjudication and refinement constraints.

The structural checks are the "SCM validation" component of the ablation:
* acyclicity of the graph of Verified/Partial propositions (a cycle that no
  single edge removal breaks is a hard violation);
* direction: a further causal claim that runs against a supported path
  (o ~> s in the graph while the claim says s -> o);
* intervention consistency: a simple linear SCM over the graph (unit-weight
  edges, signed) simulates do(s); a claim whose sign, or whose negation,
  disagrees with the simulated effect on o is a violation.
"""

import networkx as nx

from .config import CAFSettings
from .kb import PHRASE, polarity
from .verifier import CONTRADICTION, FAILED, PARTIAL, VERIFIED

ACCEPT, REFINE, REJECT, TIMEOUT = "ACCEPT", "REFINE", "REJECT", "TIMEOUT"


def caf_score(results: list[dict], st: CAFSettings) -> float:
    n = len(results)
    if n == 0:
        return 0.0
    v = sum(r["status"] == VERIFIED for r in results)
    p = sum(r["status"] == PARTIAL for r in results)
    c = sum(r["status"] == CONTRADICTION for r in results)
    return (v + st.alpha * p - st.beta * c) / n


def entailment(results: list[dict]) -> float:
    """Entailment accuracy of one output, as in the original metrics code:
    (verified + 0.5 * partial) / propositions checked; 0 when nothing was checked."""
    if not results:
        return 0.0
    v = sum(r["status"] == VERIFIED for r in results)
    p = sum(r["status"] == PARTIAL for r in results)
    return (v + 0.5 * p) / len(results)


def proposition_graph(results: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for r in results:
        if r["status"] in (VERIFIED, PARTIAL) and not r["negated"] and r["relation"] != "independent_of":
            g.add_edge(r["s_link"]["uri"], r["o_link"]["uri"], sign=polarity(r["relation"]))
    return g


def simulated_effect(g: nx.DiGraph, s: str, o: str) -> int:
    """Effect of do(s=1) versus do(s=0) on o in a linear unit-weight SCM over g
    (graph surgery: edges into s are cut). 0 means no directed path."""
    if s not in g or o not in g or not nx.has_path(g, s, o):
        return 0
    h = g.copy()
    h.remove_edges_from(list(h.in_edges(s)))
    sub = h.subgraph(nx.descendants(h, s) | {s}).copy()
    if not nx.is_directed_acyclic_graph(sub):
        total = 0  # cyclic: sum signed simple paths instead
        for path in nx.all_simple_paths(h, s, o):
            sign = 1
            for u, v in zip(path, path[1:]):
                sign *= h.edges[u, v]["sign"]
            total += sign
        return total
    x = {s: 1}
    for v in nx.topological_sort(sub):
        if v == s:
            continue
        x[v] = sum(h.edges[u, v]["sign"] * x[u] for u in sub.predecessors(v) if u in x)
    return x.get(o, 0)


def structural_violations(results: list[dict]) -> list[dict]:
    g = proposition_graph(results)
    out = []
    for cycle in nx.simple_cycles(g):
        removable = any(
            nx.is_directed_acyclic_graph(nx.restricted_view(g, [], [(u, v)]))
            for u, v in zip(cycle, cycle[1:] + cycle[:1])
        )
        out.append({"kind": "cycle", "nodes": cycle, "hard": not removable, "prop": None})
    for r in results:
        if r["status"] != FAILED or not r["s_link"] or not r["o_link"]:
            continue
        s, o = r["s_link"]["uri"], r["o_link"]["uri"]
        if s == o or r["relation"] == "independent_of":
            continue
        if r["negated"]:
            eff = simulated_effect(g, s, o)
            if eff != 0 and (eff > 0) == (polarity(r["relation"]) > 0):
                out.append({"kind": "intervention", "prop": r["prop"], "hard": False,
                            "detail": "the supported claims imply this effect"})
        elif s in g and o in g and nx.has_path(g, o, s):
            out.append({"kind": "direction", "prop": r["prop"], "hard": False,
                        "detail": "runs against the direction of the supported claims"})
        else:
            eff = simulated_effect(g, s, o)
            if eff != 0 and (eff > 0) != (polarity(r["relation"]) > 0):
                out.append({"kind": "intervention", "prop": r["prop"], "hard": False,
                            "detail": "simulated intervention gives the opposite effect"})
    return out


def _kb_statement(r: dict) -> str | None:
    rels = [q for q in r["kb_relations"] if q not in ("leads_to", "prevents")] or r["kb_relations"]
    if not rels or not r["s_link"] or not r["o_link"]:
        return None
    return f"{r['s_link']['label']} {PHRASE[rels[0]]} {r['o_link']['label']}"


def constraints_for(results: list[dict], violations: list[dict], samples_had_props: bool = False) -> list[str]:
    """ExtractConstraints (Section 4.5.3). Priority: structural > contradiction > partial."""
    out: list[str] = []
    done: set[str] = set()
    for v in violations:
        if v["prop"] and v["prop"] not in done:
            out.append(f"Do not assert that {v['prop']}: it {v['detail']}.")
            done.add(v["prop"])
        elif v["kind"] == "cycle":
            out.append("Do not state causal claims that form a cycle.")
    for r in results:
        if r["status"] == CONTRADICTION and r["prop"] not in done:
            c = f"Do not assert that {r['prop']}."
            fix = _kb_statement(r)
            if fix:
                c += f" The supported relation is: {fix}."
            out.append(c)
            done.add(r["prop"])
    for r in results:
        if r["status"] == PARTIAL and r["prop"] not in done:
            out.append(
                f"Restate the relation between '{r['subject']}' and '{r['obj']}' "
                "with a more precisely supported relation."
            )
            done.add(r["prop"])
    if not results and samples_had_props:
        out.append("State only the causal claims the premises support; "
                   "the sampled answers did not agree on any claim.")
    elif not results:
        out.append("End the answer with your causal claims as [PROP] cause | relation | effect lines.")
    return out


def adjudicate(results: list[dict], violations: list[dict], t: int, st: CAFSettings,
               samples_had_props: bool = False) -> dict:
    """Algorithm 4.2. t is the 0-based iteration index."""
    score = caf_score(results, st)
    if results and score >= st.theta and not violations:
        return {"decision": ACCEPT, "score": score, "constraints": []}
    failed = sum(r["status"] == FAILED for r in results)
    if results and failed / len(results) > st.phi_fail:
        return {"decision": REJECT, "score": score, "constraints": [], "why": "failed fraction"}
    if any(v["hard"] for v in violations):
        return {"decision": REJECT, "score": score, "constraints": [], "why": "hard violation"}
    if t + 1 >= st.iterations:
        return {"decision": TIMEOUT, "score": score, "constraints": []}
    return {"decision": REFINE, "score": score,
            "constraints": constraints_for(results, violations, samples_had_props)}
