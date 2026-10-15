"""
Generation backends. All requests of one round are sent together, so the GPU
works on a full batch.

* vllm  - recommended on the RTX 3090 (Linux or WSL2); per-request seeds.
* hf    - plain transformers, slower; works anywhere torch has a GPU.
* mock  - a scripted stand-in used only by the unit tests. Its output is not
          model output; run.py marks such runs and analyze.py refuses them.
"""

import hashlib
import os
import random
import re
from dataclasses import dataclass

from .config import DecodingSettings

# Set before vLLM is imported anywhere (run.py also imports it for the manifest).
# Batch-invariant kernels make a request's output independent of what else is in
# the batch: reruns reproduce exactly, and Full CAF and its ablations see the same
# text until their prompts actually differ (tested on the RTX 3090, vLLM 0.21).
VLLM_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    # FlashInfer's sampler JIT-compiles kernels and looks for nvcc under the pip
    # nvidia/cu13 wheel, which ships no nvcc; use vLLM's native torch sampler.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
for _k, _v in VLLM_ENV.items():
    os.environ.setdefault(_k, _v)


@dataclass
class GenRequest:
    system: str
    user: str
    n: int
    seed: int


def request_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        model: str,
        dec: DecodingSettings,
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        dtype="float16",
    ):
        from vllm import LLM

        self.model = model
        self.dec = dec
        self.llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()

    def _fmt(self, r: GenRequest) -> str:
        msgs = [
            {"role": "system", "content": r.system},
            {"role": "user", "content": r.user},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate(self, reqs: list[GenRequest]) -> list[list[str]]:
        from vllm import SamplingParams

        if not reqs:
            return []
        params = [
            SamplingParams(
                n=r.n,
                temperature=self.dec.temperature,
                top_p=self.dec.top_p,
                repetition_penalty=self.dec.repetition_penalty,
                max_tokens=self.dec.max_new_tokens,
                seed=r.seed,
            )
            for r in reqs
        ]
        outs = self.llm.generate([self._fmt(r) for r in reqs], params, use_tqdm=True)
        return [[strip_think(o.text) for o in out.outputs] for out in outs]


class HFBackend:
    name = "hf"

    def __init__(
        self, model: str, dec: DecodingSettings, batch_size: int = 8, dtype="float16"
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model
        self.dec = dec
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype) if device != "cpu" else torch.float32
        ).to(device)
        self.model.eval()

    def _fmt(self, r: GenRequest) -> str:
        msgs = [
            {"role": "system", "content": r.system},
            {"role": "user", "content": r.user},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate(self, reqs: list[GenRequest]) -> list[list[str]]:
        results: list[list[str] | None] = [None] * len(reqs)
        # group by n so one generate() call can use num_return_sequences
        order = sorted(range(len(reqs)), key=lambda i: (reqs[i].n, len(reqs[i].user)))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            for n in sorted({reqs[i].n for i in idx}):
                group = [i for i in idx if reqs[i].n == n]
                self.torch.manual_seed(reqs[group[0]].seed)
                enc = self.tok(
                    [self._fmt(reqs[i]) for i in group],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                with self.torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        do_sample=True,
                        temperature=self.dec.temperature,
                        top_p=self.dec.top_p,
                        repetition_penalty=self.dec.repetition_penalty,
                        max_new_tokens=self.dec.max_new_tokens,
                        num_return_sequences=n,
                        pad_token_id=self.tok.pad_token_id,
                    )
                gen = gen[:, enc["input_ids"].shape[1] :]
                texts = self.tok.batch_decode(gen, skip_special_tokens=True)
                for j, i in enumerate(group):
                    results[i] = [strip_think(t) for t in texts[j * n : (j + 1) * n]]
        return results  # type: ignore[return-value]


class MockBackend:
    """Scripted stand-in for tests: restates the premises as [PROP] lines with
    seeded errors (dropped, reversed or sign-flipped claims) and honours
    'Do not assert that ...' constraints. NOT a language model."""

    name = "mock"

    def __init__(self, error_rate: float = 0.3):
        self.error_rate = error_rate

    def generate(self, reqs: list[GenRequest]) -> list[list[str]]:
        out = []
        for r in reqs:
            samples = []
            for k in range(r.n):
                rng = random.Random(r.seed * 1000 + k)
                samples.append(self._one(r, rng))
            out.append(samples)
        return out

    def _one(self, r: GenRequest, rng: random.Random) -> str:
        banned = set(
            re.findall(
                r"Do not assert that (.+?)(?::|\.(?:\s|$))", r.system + "\n" + r.user
            )
        )
        lines = []
        task = (
            r.user.split("\n\n", 1)[1]
            if r.user.startswith("IMPORTANT CONSTRAINTS")
            else r.user
        )
        prem = re.findall(r"^\s*(?:-|\d+\.)\s*(.+?)\s*$", task, flags=re.M)
        rel_re = re.compile(
            r"^(.+?) (leads to|causes|implies that|enables|prevents|requires|triggers) (.+)$"
        )
        first = last = None
        for p in prem:
            m = rel_re.match(p)
            if not m:
                continue
            s, rel, o = m.group(1), m.group(2).replace(" that", ""), m.group(3)
            first = first or s
            last = o
            x = rng.random()
            if x < self.error_rate / 3:
                continue
            if x < 2 * self.error_rate / 3:
                s, o = o, s
            elif x < self.error_rate:
                rel = "prevents" if rel != "prevents" else "causes"
            if any(b.startswith(s + " ") and b.endswith(" " + o) for b in banned):
                continue
            lines.append(f"[PROP] {s} | {rel} | {o}")
        if first and last:
            lines.append(f"[PROP] {first} | leads to | {last}")
        return "Answer: the chain of effects follows from the premises.\n" + "\n".join(
            lines
        )


def make_backend(kind: str, model: str, dec: DecodingSettings, **kw):
    if kind == "vllm":
        return VLLMBackend(
            model,
            dec,
            **{
                k: v
                for k, v in kw.items()
                if k in ("gpu_memory_utilization", "max_model_len")
            },
        )
    if kind == "hf":
        return HFBackend(
            model, dec, **{k: v for k, v in kw.items() if k in ("batch_size",)}
        )
    if kind == "mock":
        return MockBackend()
    raise ValueError(kind)
