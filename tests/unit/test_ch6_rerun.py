"""
Chapter 6 re-run harness: knowledge base, parser, verifier, Deterministic
Executive, and an end-to-end mock run (no GPU, no network: a bag-of-words
encoder stands in for Sentence-BERT).
"""

import json

import numpy as np
import pytest

from experiments.ch6_rerun import analyze, run
from experiments.ch6_rerun.config import FULL, CAFSettings
from experiments.ch6_rerun.executive import (
    ACCEPT,
    REFINE,
    REJECT,
    TIMEOUT,
    adjudicate,
    caf_score,
    constraints_for,
    simulated_effect,
    structural_violations,
)
from experiments.ch6_rerun.kb import build_kb
from experiments.ch6_rerun.linker import Encoder, EntityLinker
from experiments.ch6_rerun.props import Proposition, consensus, parse_response
from experiments.ch6_rerun.verifier import (
    CONTRADICTION,
    FAILED,
    PARTIAL,
    VERIFIED,
    Verifier,
)
from experiments.synthetic_dataset import CausalChain, LogicalStep, RelationType


class BagOfWords:
    """Deterministic stand-in for SentenceTransformer.encode."""

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in t.lower().split():
                v[hash(w.rstrip("s")) % 256] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.stack(out)


def chain(*steps):
    return CausalChain(
        chain_id="c1",
        steps=[
            LogicalStep(a, b, RelationType(r), step_number=i)
            for i, (a, r, b) in enumerate(steps)
        ],
        domain="physics",
        complexity_score=0.5,
        ground_truth_entailments=[],
    )


@pytest.fixture
def kb():
    return build_kb(
        chain(
            ("energy added", "leads_to", "temperature rises"),
            ("temperature rises", "causes", "current decreases"),
            ("current decreases", "prevents", "acceleration occurs"),
        )
    )


@pytest.fixture
def verifier():
    return Verifier(EntityLinker(Encoder("bow", model=BagOfWords()), threshold=0.7))


def P(s, r, o, neg=False):
    return Proposition(s, r, o, neg)


def test_kb_holds_stated_and_entailed_relations(kb):
    def e(text):
        return kb.by_norm[text]

    assert kb.predicates(e("energy added"), e("temperature rises")) == {"leads_to"}
    assert kb.predicates(e("energy added"), e("current decreases")) == {"leads_to"}
    # + then + then - : net negative
    assert kb.predicates(e("energy added"), e("acceleration occurs")) == {"prevents"}
    assert kb.predicates(e("temperature rises"), e("energy added")) == set()


def test_verification_outcomes(kb, verifier):
    res = verifier.verify(
        [
            P("energy added", "leads_to", "temperature rises"),  # exact
            P(
                "the temperature rises", "causes", "current decreases"
            ),  # exact after normalisation
            P("energy added", "causes", "current decreases"),  # related predicate
            P(
                "energy added", "causes", "acceleration occurs"
            ),  # opposing (net negative)
            P("temperature rises", "causes", "energy added"),  # reversed: no support
            P("unicorns", "causes", "energy added"),  # unlinked
            P(
                "energy added", "causes", "temperature rises", True
            ),  # negation of a supported link
        ],
        kb,
        FULL,
    )
    assert [r["status"] for r in res] == [
        VERIFIED,
        VERIFIED,
        PARTIAL,
        CONTRADICTION,
        FAILED,
        FAILED,
        CONTRADICTION,
    ]


def test_partial_matching_switch(kb, verifier):
    st = CAFSettings(partial_matching=False)
    [r] = verifier.verify([P("energy added", "causes", "current decreases")], kb, st)
    assert r["status"] == FAILED


def test_entity_linking_switch(kb, verifier):
    mention = "energy added quickly"  # not an exact label; close in embedding space
    on = verifier.verify([P(mention, "leads_to", "temperature rises")], kb, FULL)[0]
    off = verifier.verify(
        [P(mention, "leads_to", "temperature rises")],
        kb,
        CAFSettings(entity_linking=False),
    )[0]
    assert on["status"] == VERIFIED and on["s_link"]["method"] == "embedding"
    assert off["status"] == FAILED


def test_structural_checks_catch_reversed_claim(kb, verifier):
    res = verifier.verify(
        [
            P("energy added", "leads_to", "temperature rises"),
            P("temperature rises", "causes", "current decreases"),
            P(
                "current decreases", "causes", "energy added"
            ),  # closes a loop against supported path
        ],
        kb,
        FULL,
    )
    v = structural_violations(res)
    assert [x["kind"] for x in v] == ["direction"]
    assert any(
        "Do not assert that current decreases causes energy added" in c
        for c in constraints_for(res, v)
    )


def test_simulated_intervention_sign():
    import networkx as nx

    g = nx.DiGraph()
    g.add_edge("a", "b", sign=1)
    g.add_edge("b", "c", sign=-1)
    assert simulated_effect(g, "a", "c") == -1
    assert simulated_effect(g, "c", "a") == 0


def test_adjudication_paths(kb, verifier):
    good = verifier.verify(
        [P("energy added", "leads_to", "temperature rises")], kb, FULL
    )
    assert adjudicate(good, [], 0, FULL)["decision"] == ACCEPT
    mixed = verifier.verify(
        [
            P("energy added", "leads_to", "temperature rises"),
            P("energy added", "causes", "acceleration occurs"),
        ],
        kb,
        FULL,
    )
    a = adjudicate(mixed, [], 0, FULL)
    assert a["decision"] == REFINE and a["constraints"][0].startswith(
        "Do not assert that energy added causes"
    )
    assert "energy added prevents acceleration occurs" in a["constraints"][0]
    assert adjudicate(mixed, [], 4, FULL)["decision"] == TIMEOUT
    assert (
        adjudicate(mixed, [], 0, CAFSettings(iterative_feedback=False))["decision"]
        == TIMEOUT
    )
    unsupported = verifier.verify([P("unicorns", "causes", "rain")], kb, FULL)
    assert adjudicate(unsupported, [], 0, FULL)["decision"] == REJECT
    assert caf_score(mixed, FULL) == pytest.approx((1 - 0.5) / 2)


def test_parser_structured_and_fallback():
    text = """The temperature rises, so current falls.
[PROP] energy added | leads to | temperature rises
[PROP] temperature rises | does not cause | current decreases
**[PROP]** current decreases | ultimately leads to | acceleration occurs
[PROP] X | flibbers | Y"""
    ps = parse_response(text)
    assert [(p.subject, p.relation, p.obj, p.negated) for p in ps] == [
        ("energy added", "leads_to", "temperature rises", False),
        ("temperature rises", "causes", "current decreases", True),
        ("current decreases", "leads_to", "acceleration occurs", False),
    ]
    free = parse_response(
        "Energy added causes temperature rises. The opposite of heat occurs when ice forms."
    )
    assert [(p.subject, p.relation, p.obj) for p in free] == [
        ("Energy added", "causes", "temperature rises"),
        ("ice forms", "prevents", "heat"),
    ]
    assert all(p.source == "fallback" for p in free)


def test_parser_passive_forms_swap_direction():
    ps = parse_response(
        "[PROP] acceleration occurs | is caused by | current decreases\n"
        "[PROP] flooding | is not prevented by | dams"
    )
    assert [(p.subject, p.relation, p.obj, p.negated) for p in ps] == [
        ("current decreases", "causes", "acceleration occurs", False),
        ("dams", "prevents", "flooding", True),
    ]
    [p] = parse_response("Crop damage results from flooding occurs.")
    assert (p.subject, p.relation, p.obj) == (
        "flooding occurs",
        "leads_to",
        "Crop damage",
    )


def test_parser_relation_words_inside_entity_names():
    [p] = parse_response("Aggregate demand increases does not cause price rises.")
    assert (p.subject, p.relation, p.negated, p.obj) == (
        "Aggregate demand increases",
        "causes",
        True,
        "price rises",
    )
    [p] = parse_response("Therefore, alarm triggers causes evacuation begins.")
    assert (p.subject, p.obj) == ("alarm triggers", "evacuation begins")


def test_consensus_keeps_recurring_propositions():
    a, b, c = P("x", "causes", "y"), P("y", "causes", "z"), P("q", "causes", "r")
    kept, best, votes = consensus([[a, b], [a, b, c], [a], [a, b], [b]], 0.6)
    assert {p.key() for p in kept} == {a.key(), b.key()}
    assert best == 0


def test_end_to_end_mock_run_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.ch6_rerun.run.Encoder",
        lambda *a, **k: Encoder("bow", model=BagOfWords()),
    )
    out = tmp_path / "r"
    argv = [
        "--backend",
        "mock",
        "--chains",
        "5",
        "--seeds",
        "1",
        "2",
        "--out",
        str(out),
        "--systems",
        "vanilla",
        "rag",
        "caf",
        "caf_no_scm",
    ]
    run.main(argv)
    out = tmp_path / "r_MOCK"
    lines = (out / "seed_1" / "caf.jsonl").read_text().splitlines()
    assert len(lines) == 20  # 5 chains x (original + 3 perturbations)
    rec = json.loads(lines[0])
    assert rec["iterations"][0]["samples"] and "eval_results" in rec["final"]
    run.main(argv)  # resume: nothing re-run, nothing duplicated
    assert len((out / "seed_1" / "caf.jsonl").read_text().splitlines()) == 20
    with pytest.raises(SystemExit):
        analyze.main([str(out)])  # mock results are refused
    analyze.main([str(out), "--allow-mock"])
    summary = (out / "summary.md").read_text()
    assert "Full CAF" in summary and "Wilcoxon" in summary


def test_parser_reads_unmarked_pipe_lines():
    ps = parse_response(
        "[demand increases] causes | relation | [price rises]\n"
        "[price rises] leads to | currency strengthens"
    )
    assert [(p.subject, p.relation, p.obj) for p in ps] == [
        ("demand increases", "causes", "price rises"),
        ("price rises", "leads_to", "currency strengthens"),
    ]


def test_constraints_are_prepended_to_the_user_prompt():
    from experiments.ch6_rerun.prompts import system_message, with_constraints

    u = with_constraints("TASK", ["Do not assert that a causes b."])
    assert u.startswith("IMPORTANT CONSTRAINTS") and u.endswith("TASK")
    assert "Do not assert" not in system_message()
    assert with_constraints("TASK", []) == "TASK"


def test_embedding_links_must_preserve_polarity_and_logic():
    from experiments.ch6_rerun.linker import polarity_signature as sig

    same = [
        ("an increase in temperature", "temperature rises"),
        ("current falls", "current decreases"),
        ("energy being added", "energy added"),
        ("A and B hold", "a and b are true"),
    ]
    different = [
        ("aggregate demand decreases", "aggregate demand increases"),
        ("no temperature rise", "temperature rises"),
        ("lack of energy added", "energy added"),
        ("P and Q hold", "P or Q holds"),
        ("P holds", "not-P holds"),
        ("P holds", "Q holds"),
    ]
    assert all(sig(a) == sig(b) for a, b in same)
    assert all(sig(a) != sig(b) for a, b in different)


def test_negated_mention_is_not_linked_to_positive_entity(kb, verifier):
    [r] = verifier.verify(
        [P("no temperature rise", "causes", "current decreases")], kb, FULL
    )
    assert r["s_link"] is None and r["status"] == FAILED


def test_parser_formats_seen_in_qwen_pilot():
    cases = {
        "[Q is true] true | implies | R follows": ("Q is true", "implies", "R follows"),
        "[if P then Q, P holds] implies | that | Q holds": (
            "if P then Q, P holds",
            "implies",
            "Q holds",
        ),
        "[PROP] if P then Q, P holds | implies | Q is true": (
            "if P then Q, P holds",
            "implies",
            "Q is true",
        ),
        "If energy is added, then temperature rises.": (
            "energy is added",
            "leads_to",
            "temperature rises",
        ),
    }
    for text, want in cases.items():
        [p] = parse_response(text)
        assert (p.subject, p.relation, p.obj) == want, text


def test_double_negation_mentions_normalise_to_the_event():
    from experiments.ch6_rerun.kb import norm_text

    assert (
        norm_text("It is not the case that alarm triggers does not occur.")
        == "alarm triggers"
    )


def test_compound_mentions_do_not_embedding_link():
    from experiments.ch6_rerun.linker import polarity_signature as sig

    assert sig("if P then Q, P holds implies that Q holds") != sig(
        "if P then Q, P holds"
    )
    assert sig("alarm triggers") == sig("Alarm triggers")


def test_parser_column_variants():
    cases = {
        "[PROP] energy added does not cause | acceleration occurs": (
            "energy added",
            "causes",
            True,
            "acceleration occurs",
        ),
        "[investment grows] does not cause | prevents | price rises": (
            "investment grows",
            "prevents",
            False,
            "price rises",
        ),
        "[borrowing drops] causes | leads to | price rises": (
            "borrowing drops",
            "leads_to",
            False,
            "price rises",
        ),
        "[C can be derived] implies | implies | [R follows]": (
            "C can be derived",
            "implies",
            False,
            "R follows",
        ),
        "[R] Q is true | implies | R follows": (
            "Q is true",
            "implies",
            False,
            "R follows",
        ),
    }
    for text, want in cases.items():
        [p] = parse_response(text)
        assert (p.subject, p.relation, p.negated, p.obj) == want, text


def test_negation_parity_and_add_remove():
    from experiments.ch6_rerun.kb import norm_text
    from experiments.ch6_rerun.linker import polarity_signature as sig

    assert sig("not-Q is false") != sig("not-Q follows")
    assert sig("not-Q does not hold") != sig("not-Q follows")
    assert sig("Q is not true") == sig("not-Q follows")
    assert sig("energy removed") != sig("energy added")
    assert (
        norm_text("It is the case that acceleration occurs does occur")
        == "acceleration occurs"
    )
