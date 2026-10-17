from __future__ import annotations

import re


SPATIAL_QUERY = re.compile(
    r"^\s*(who|how many)\b|"
    r"\b(color|colour|name|named|song|word|number|score|text|wearing|dressed|shown on the screen)\b|"
    r"\b(say|says|said|claim|claims|claimed|report|reports|reported|call|calls|called)\b",
    re.IGNORECASE,
)

TEMPORAL_QUERY = re.compile(
    r"\b(before|after|then|next|subsequent|subsequently|prior|following|first|last)\b|"
    r"\bwhat happens?\b|"
    r"\b(draw|draws|drew|pick|picks|picked|take|takes|took|put|puts|move|moves|moved|"
    r"enter|enters|entered|leave|leaves|left|fall|falls|fell|turn|turns|turned|"
    r"become|becomes|became|change|changes|changed|start|starts|started|end|ends|ended)\b",
    re.IGNORECASE,
)


def choose_representation(question: str) -> str:
    """Choose a frozen evidence topology without inspecting outcomes."""
    if SPATIAL_QUERY.search(question):
        return "images"
    if TEMPORAL_QUERY.search(question):
        return "span_videos"
    return "images"
