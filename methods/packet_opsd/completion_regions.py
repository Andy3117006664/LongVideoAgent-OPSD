"""Exact-ID supervision masks for opt-in token-KL ablations.

This module never samples, calls a tool or infers action correctness. Full KL
keeps the original path. Non-full masks require an exact full-string tokenizer
round trip with offsets; independently retokenizing JSON substrings is unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


class CompletionRegionError(ValueError):
    """Reject the selected record rather than silently reverting to full KL."""


def require(value, code):
    if not value:
        raise CompletionRegionError(code)


@dataclass(frozen=True)
class CompletionRegion:
    region: str
    mask: list[bool] | None
    full_tokens: int
    selected_tokens: int
    boundary_tokens_excluded: int = 0


def _space(text, at):
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _members(text, start):
    decoder = json.JSONDecoder()
    require(start < len(text) and text[start] == "{", "tool-json-object")
    at, spans = _space(text, start + 1), {}
    require(at < len(text) and text[at] != "}", "tool-json-empty")
    while True:
        key, end = decoder.raw_decode(text, at)
        require(isinstance(key, str) and key not in spans, "tool-json-duplicate-or-invalid-key")
        at = _space(text, end)
        require(at < len(text) and text[at] == ":", "tool-json-colon")
        at = _space(text, at + 1)
        value, end = decoder.raw_decode(text, at)
        spans[key] = (at, end, value)
        at = _space(text, end)
        require(at < len(text), "tool-json-incomplete")
        if text[at] == "}":
            return spans, at + 1
        require(text[at] == ",", "tool-json-comma")
        at = _space(text, at + 1)


def action_character_spans(text, terminal_texts):
    """Find one complete actual tool call after the last thinking delimiter.

    JSON raw decoding handles nested objects/escaped braces. The original
    harness parses the first call in the whole completion, so any extra opener
    or closer is refused, even a quoted example in the thinking section. Never
    skip a malformed first call to supervise a later valid one. A literal
    thinking delimiter is a format boundary, not a semantic reasoning verdict.
    """
    terminals = tuple(sorted(set(terminal_texts), key=len, reverse=True))
    require(all(isinstance(s, str) and s for s in terminals), "terminal-text")
    require(text.count("<tool_call>") == 1 and text.count("</tool_call>") == 1,
            "exactly-one-tool-marker-pair")
    close_think = text.rfind("</think>")
    zone = close_think + len("</think>") if close_think >= 0 else 0
    candidates, offset = [], zone
    while True:
        begin = text.find("<tool_call>", offset)
        if begin < 0:
            break
        offset = begin + len("<tool_call>")
        try:
            spans, end = _members(text, _space(text, offset))
            require(set(spans) == {"name", "arguments"}, "tool-json-members")
            require(spans["name"][2] in {"visual_inspect", "visual_retrieve"}, "tool-name")
            require(isinstance(spans["arguments"][2], dict), "tool-arguments-object")
            end = _space(text, end)
            require(text.startswith("</tool_call>", end), "tool-close")
            candidates.append((begin, end + len("</tool_call>"), spans["arguments"][:2]))
        except (ValueError, TypeError):
            pass
    require(len(candidates) == 1, "exactly-one-complete-tool-call")
    begin, end, arguments = candidates[0]
    if close_think >= 0:
        require(not text[zone:begin].strip(), "text-after-think-before-tool")
    at = end
    while at < len(text):
        if text[at].isspace():
            at = _space(text, at)
            continue
        terminal = next((s for s in terminals if text.startswith(s, at)), None)
        require(terminal is not None, "nonterminal-tool-suffix")
        at += len(terminal)
    return {"action": (begin, end), "arguments": arguments}


def completion_region(tokenizer, completion, *, region: str, eos_token_ids) -> CompletionRegion:
    """Select only whole original tokens, retaining the full original N divisor."""
    require(region in {"full", "action", "arguments"}, "unknown-kl-region")
    ids = list(completion.token_ids)
    require(ids and all(type(x) is int and x >= 0 for x in ids), "original-token-ids-required")
    if region == "full":
        # Do not impose new parsing/tokenizer constraints on the old default.
        return CompletionRegion(region, None, len(ids), len(ids))
    require(completion.complete and getattr(completion, "stop_reason", None) != "token_cap",
            "incomplete-kl-completion")
    require(getattr(tokenizer, "is_fast", False), "fast-tokenizer-required")
    try:
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        require(list(encoded["input_ids"]) == ids, "full-id-roundtrip-mismatch")
        offsets = list(encoded["offset_mapping"])
        require(len(offsets) == len(ids), "token-offset-length")
        eos = set(eos_token_ids)
        terminal = tuple(tokenizer.decode([x], skip_special_tokens=False,
                         clean_up_tokenization_spaces=False) for x in eos)
        left, right = action_character_spans(text, terminal)[region]
        mask, boundary = [], 0
        for token_id, span in zip(ids, offsets):
            require(isinstance(span, (list, tuple)) and len(span) == 2, "token-offset-format")
            start, end = span
            require(type(start) is int and type(end) is int and 0 <= start <= end <= len(text),
                    "token-offset-bounds")
            if token_id in eos:
                mask.append(False)
                continue
            require(start < end, "zero-width-nonstop-token")
            selected = left <= start and end <= right
            boundary += int(not selected and start < right and end > left)
            mask.append(selected)
        require(any(mask), "empty-kl-region")
        return CompletionRegion(region, mask, len(ids), sum(mask), boundary)
    except CompletionRegionError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, NotImplementedError) as exc:
        raise CompletionRegionError("invalid-kl-offset-or-tool-format") from exc
