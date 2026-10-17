from __future__ import annotations

import importlib
import os
from typing import Dict, Type

from videoseal.tools.base import Tool
from videoseal.tools.visual_tools import VisualInspectAliasTool, VisualRetrieveAliasTool


def build_tool_map() -> Dict[str, Type[Tool]]:
    # Optional project-local intervention hook. The released baseline remains
    # byte-for-byte behaviorally identical when this variable is unset.
    spec = str(os.getenv("VIDEOSEAL_VISUAL_RETRIEVE_CLASS") or "").strip()
    if spec:
        module_name, separator, class_name = spec.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError("VIDEOSEAL_VISUAL_RETRIEVE_CLASS must be module:Class")
        custom = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(custom, type) or not issubclass(custom, Tool):
            raise TypeError(f"{spec} is not a Tool subclass")
    else:
        custom = VisualRetrieveAliasTool
    inspect_spec = str(os.getenv("VIDEOSEAL_VISUAL_INSPECT_CLASS") or "").strip()
    if inspect_spec:
        inspect_module, inspect_separator, inspect_name = inspect_spec.partition(":")
        if not inspect_separator or not inspect_module or not inspect_name:
            raise ValueError("VIDEOSEAL_VISUAL_INSPECT_CLASS must be module:Class")
        inspect_custom = getattr(importlib.import_module(inspect_module), inspect_name)
        if not isinstance(inspect_custom, type) or not issubclass(inspect_custom, Tool):
            raise TypeError(f"{inspect_spec} is not a Tool subclass")
    else:
        inspect_custom = VisualInspectAliasTool
    return {
        "visual_retrieve": custom,
        "visual_inspect": inspect_custom,
    }
