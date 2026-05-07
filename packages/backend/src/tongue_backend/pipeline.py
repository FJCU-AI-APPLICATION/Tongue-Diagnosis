"""End-to-end inference pipeline: bytes → heads → LLM → AnalyzeResponse."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tongue_ai.inference import run_all
from tongue_ai.preprocessing import decode_bgr
from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult

from tongue_backend.llm.user_message import build as build_user_message


DISCLAIMER = "此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。"


class _LLMClientProto(Protocol):
    def run(self, *, system: str, user: str, config: dict[str, Any]) -> str: ...


@dataclass
class AnalyzeResponse:
    user_message: str
    heads: list[HeadResult]
    comment: str
    disclaimer: str
    category_map: dict[str, dict[str, str]]
    timing_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "heads": [
                {
                    "task": h.task,
                    "head_type": h.head_type,
                    "predictions": [{"label": p.label, "score": p.score} for p in h.predictions],
                    "error": h.error,
                }
                for h in self.heads
            ],
            "comment": self.comment,
            "disclaimer": self.disclaimer,
            "category_map": self.category_map,
            "timing_ms": self.timing_ms,
        }


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def analyze(
    *,
    image_bytes: bytes,
    registry: Registry,
    prompt: str,
    llm_config: dict[str, Any],
    llm_client: _LLMClientProto,
) -> AnalyzeResponse:
    t0 = time.perf_counter()
    image = decode_bgr(image_bytes)
    t_decode = _ms_since(t0)

    t1 = time.perf_counter()
    heads = run_all(image, registry)
    t_infer = _ms_since(t1)

    user_msg = build_user_message(heads, registry.category_map)

    t2 = time.perf_counter()
    comment = llm_client.run(system=prompt, user=user_msg, config=llm_config)
    t_llm = _ms_since(t2)

    return AnalyzeResponse(
        user_message=user_msg,
        heads=heads,
        comment=comment,
        disclaimer=DISCLAIMER,
        category_map=registry.category_map,
        timing_ms={
            "decode": t_decode,
            "infer": t_infer,
            "llm": t_llm,
            "total": _ms_since(t0),
        },
    )
