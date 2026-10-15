"""
Per-chain knowledge base (an in-memory RDF graph queried with SPARQL).

For each chain the knowledge base holds what the chain states and what it
entails, which makes it correct by construction (Section 4.6):

* every stated step as  <antecedent> caf:<relation> <consequent>;
* for every ordered pair joined by a directed path, an entailed relation
  caf:leads_to (net positive path) or caf:prevents (net negative path).

The injected contradiction is never added: it is the error to be caught.
"""

import itertools
import re
from dataclasses import dataclass

import networkx as nx
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDFS

from experiments.synthetic_dataset import CausalChain

EX = Namespace("http://caf.local/entity/")
CAF = Namespace("http://caf.local/relation/")

RELATIONS = ["causes", "leads_to", "enables", "implies", "requires", "influences", "prevents"]
NEGATIVE = {"prevents"}

PHRASE = {
    "causes": "causes",
    "leads_to": "leads to",
    "enables": "enables",
    "implies": "implies",
    "requires": "requires",
    "influences": "influences",
    "prevents": "prevents",
}


def polarity(relation: str) -> int:
    return -1 if relation in NEGATIVE else 1


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def norm_text(text: str) -> str:
    """Normalisation used for exact label matching and proposition keys."""
    t = text.lower().strip()
    t = re.sub(r"[*_`\"'“”‘’]", " ", t)
    t = re.sub(r"^(given|therefore|thus|hence|so|then|finally|ultimately)[:,]?\s+", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[\s]+", " ", t)
    return t.strip(" .,;:!?()[]")


@dataclass
class ChainKB:
    chain_id: str
    graph: Graph
    labels: dict[str, str]  # uri -> label
    by_norm: dict[str, str]  # normalised label -> uri
    facts: list[str]  # natural-language statements, for the RAG baselines
    structure: nx.DiGraph  # stated steps, signed

    def predicates(self, s_uri: str, o_uri: str) -> set[str]:
        """SPARQL: every caf: relation the knowledge base holds from s to o."""
        q = (
            "SELECT DISTINCT ?p WHERE { ?s ?p ?o . "
            f'FILTER(STRSTARTS(STR(?p), "{CAF}")) }}'
        )
        rows = self.graph.query(q, initBindings={"s": URIRef(s_uri), "o": URIRef(o_uri)})
        return {str(r.p)[len(str(CAF)) :] for r in rows}

    def ask(self, s_uri: str, relation: str, o_uri: str) -> bool:
        q = "ASK { ?s ?p ?o }"
        res = self.graph.query(
            q,
            initBindings={"s": URIRef(s_uri), "p": CAF[relation], "o": URIRef(o_uri)},
        )
        return bool(res.askAnswer)


def build_kb(chain: CausalChain) -> ChainKB:
    g = Graph()
    g.bind("ex", EX)
    g.bind("caf", CAF)
    labels: dict[str, str] = {}
    structure = nx.DiGraph()

    def node(text: str) -> str:
        uri = str(EX[slug(text)])
        if uri not in labels:
            labels[uri] = text
            g.add((URIRef(uri), RDFS.label, Literal(text, lang="en")))
        return uri

    facts = []
    for step in chain.steps:
        s, o = node(step.antecedent), node(step.consequent)
        rel = step.relation.value
        g.add((URIRef(s), CAF[rel], URIRef(o)))
        facts.append(step.to_natural_language())
        if s != o:
            structure.add_edge(s, o, relation=rel, sign=polarity(rel))

    # Entailed relations: net sign of every directed path between two nodes.
    for a, b in itertools.permutations(structure.nodes, 2):
        signs = set()
        for path in nx.all_simple_paths(structure, a, b):
            sign = 1
            for u, v in zip(path, path[1:]):
                sign *= structure.edges[u, v]["sign"]
            signs.add(sign)
        for sign in signs:
            rel = "leads_to" if sign > 0 else "prevents"
            g.add((URIRef(a), CAF[rel], URIRef(b)))
            if not structure.has_edge(a, b):
                facts.append(f"{labels[a]} {PHRASE[rel]} {labels[b]}")

    by_norm = {norm_text(label): uri for uri, label in labels.items()}
    return ChainKB(chain.chain_id, g, labels, by_norm, facts, structure)
