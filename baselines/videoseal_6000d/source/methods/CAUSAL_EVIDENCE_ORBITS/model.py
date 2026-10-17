from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Mapping
from urllib import request

from .codebooks import encode_options
from .schema import Distribution, LETTERS


def _image_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _distribution(body: Mapping[str, Any]) -> Distribution:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    raw = str(message.get("content") or "").strip()
    prediction = raw[:1].upper() if raw[:1].upper() in LETTERS else ""
    values: dict[str, float] = {}
    logprobs = choice.get("logprobs") if isinstance(choice.get("logprobs"), Mapping) else {}
    for position in (logprobs.get("content") or [])[:3]:
        if not isinstance(position, Mapping):
            continue
        candidates = [position, *(position.get("top_logprobs") or [])]
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            token = str(candidate.get("token") or "").strip().upper()
            if token in LETTERS:
                values[token] = max(values.get(token, -math.inf), float(candidate["logprob"]))
        if values:
            break
    if not values:
        raise ValueError("A-D token distribution unavailable")
    floor = min(values.values()) - 8.0
    logits = {letter: values.get(letter, floor) for letter in LETTERS}
    maximum = max(logits.values())
    weights = {letter: math.exp(value - maximum) for letter, value in logits.items()}
    total = sum(weights.values())
    probabilities = {letter: value / total for letter, value in weights.items()}
    return Distribution(prediction or max(probabilities, key=probabilities.get), probabilities, raw)


class VisualMeasurementClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local-vllm",
        timeout: float = 360.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def prompt(stem: str, options: dict[str, str], token_to_semantic: dict[str, str]) -> str:
        return f"""Answer the multiple-choice question using only the supplied timestamped video frames.
If the frames are insufficient, choose the option best supported by visible evidence; do not use outside knowledge.

Question: {stem}
{encode_options(options, token_to_semantic)}

Return exactly one token: A, B, C, or D."""

    def measure(
        self,
        *,
        stem: str,
        options: dict[str, str],
        token_to_semantic: dict[str, str],
        frames: list[Path],
        timestamps: list[float],
    ) -> Distribution:
        if len(frames) != len(timestamps) or not frames:
            raise ValueError("frames and timestamps must be non-empty and aligned")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self.prompt(stem, options, token_to_semantic)}
        ]
        for frame, timestamp in zip(frames, timestamps):
            content.append({"type": "text", "text": f"[t={timestamp:.3f}s]"})
            content.append(
                {"type": "image_url", "image_url": {"url": _image_url(frame), "detail": "low"}}
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return _distribution(json.loads(response.read().decode("utf-8")))

