"""
Prompts. Every system gets the same system message and the same task prompt;
the baselines add only their own wrapper (taken from experiments/baselines.py),
and CAF adds only its refinement constraints.
"""

import numpy as np

from .kb import ChainKB
from .linker import Encoder

SYSTEM = """You are a careful assistant that reasons about cause and effect.
Answer the question using the information given. End your answer with the causal claims you are making, one per line, in exactly this form:
[PROP] cause | relation | effect
Use one of these relations: causes, leads to, enables, implies, requires, prevents, influences. Write a negated claim with "does not", for example "does not cause". Use the wording of the premises for causes and effects.

Example of the required ending:
[PROP] heavy rain | causes | flooding occurs
[PROP] flooding occurs | prevents | road access"""

FORMAT_REMINDER = "\n\nFinish with your [PROP] lines."


def system_message(constraints: list[str] | None = None) -> str:
    msg = SYSTEM
    if constraints:
        msg += "\n\nIMPORTANT CONSTRAINTS to follow:"
        for i, c in enumerate(constraints, 1):
            msg += f"\n{i}. {c}"
    return msg


def cot_wrapper(prompt: str, num_steps: int = 3) -> str:
    return f"""Let's approach this step-by-step:

{prompt}

Please think through this carefully by breaking it down into {num_steps} clear reasoning steps:
1. First, identify the key elements and relationships
2. Then, analyze each causal link and verify logical consistency
3. Finally, provide your conclusion based on the reasoning

Show your reasoning for each step."""


def rag_wrapper(prompt: str, facts: list[str]) -> str:
    if not facts:
        return prompt
    out = "Based on the following relevant facts:\n\n"
    out += "".join(f"{i}. {f}\n" for i, f in enumerate(facts, 1))
    out += f"\nNow, answer the following:\n{prompt}\n"
    out += "\nProvide a response grounded in the facts above."
    return out


def rag_cot_wrapper(prompt: str, facts: list[str]) -> str:
    out = "Based on the following relevant facts:\n\n"
    out += "".join(f"{i}. {f}\n" for i, f in enumerate(facts, 1))
    out += f"\n{prompt}\n\n"
    out += "Let's reason through this step-by-step using the facts above:\n"
    out += "1. First, identify which facts are most relevant\n"
    out += "2. Then, analyze how they connect logically\n"
    out += "3. Finally, draw a conclusion grounded in the facts\n"
    return out


def retrieve(prompt: str, kb: ChainKB, encoder: Encoder, k: int) -> list[str]:
    """Dense retrieval: the k knowledge-base statements most similar to the prompt."""
    if not kb.facts:
        return []
    mat = encoder.encode(kb.facts)
    q = encoder.encode([prompt])[0]
    order = np.argsort(-(mat @ q), kind="stable")[:k]
    return [kb.facts[i] for i in order]


def user_message(style: str, prompt: str, facts: list[str] | None) -> str:
    """The task prompt in the system's wrapper, with the same closing reminder for all."""
    if style == "plain":
        body = prompt
    elif style == "cot":
        body = cot_wrapper(prompt)
    elif style == "rag":
        body = rag_wrapper(prompt, facts or [])
    elif style == "rag_cot":
        body = rag_cot_wrapper(prompt, facts or [])
    else:
        raise ValueError(style)
    return body + FORMAT_REMINDER
