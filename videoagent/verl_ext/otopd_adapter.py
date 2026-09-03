"""Small, fail-closed helpers for actor LoRA snapshots used by OT-OPSD.

The trainer keeps the policy in FSDP shards.  This module deliberately only
handles the trainable LoRA tensors after a worker has gathered them; it never
loads or merges a full language model.  A snapshot directory is considered
usable only after ``READY`` has been written last.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert OmegaConf/PEFT values to plain JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    # OmegaConf containers expose items()/iterables but may not be dict/list.
    try:
        if hasattr(value, "items"):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
            return [_jsonable(item) for item in value]
    except Exception:
        pass
    return str(value)


def normalise_lora_key(name: str) -> str:
    """Map FSDP/PEFT names to the standard PEFT adapter key spelling."""

    key = str(name).replace("_fsdp_wrapped_module.", "")
    # PEFT state dicts produced by some versions include the active adapter
    # name (``.default``); saved adapter files omit that component.
    key = key.replace(".default.weight", ".weight").replace(".default.bias", ".bias")
    return key


def _ready(path: Path) -> bool:
    return path.is_dir() and (path / "READY").is_file() and (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def validate_adapter_dir(path: str | os.PathLike[str]) -> Path:
    """Return a resolved adapter directory, rejecting incomplete snapshots."""

    resolved = Path(path).expanduser().resolve()
    if not _ready(resolved):
        raise FileNotFoundError(f"adapter snapshot is missing READY/config/weights: {resolved}")
    return resolved


def adapter_sha256(path: str | os.PathLike[str]) -> str:
    """Hash adapter bytes and config for provenance in scorer records."""

    resolved = validate_adapter_dir(path)
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        candidate = resolved / name
        if candidate.is_file():
            digest.update(name.encode("utf-8"))
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()[:24]


def _atomic_dir(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=str(path.parent)))
    return path, tmp


def _write_ready_snapshot(
    target_dir: Path,
    *,
    state: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one complete adapter directory and atomically publish it."""

    from safetensors.torch import save_file

    target, tmp = _atomic_dir(target_dir)
    try:
        serialised = {normalise_lora_key(key): tensor.detach().cpu().contiguous() for key, tensor in state.items()}
        if not serialised or not all("lora_" in key.lower() for key in serialised):
            raise ValueError("snapshot must contain only non-empty LoRA tensors")
        if len(serialised) != len(state):
            raise ValueError("duplicate LoRA keys after PEFT/FSDP normalisation")
        save_file(serialised, str(tmp / "adapter_model.safetensors"))
        (tmp / "adapter_config.json").write_text(
            json.dumps(_jsonable(config), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if metadata is not None:
            (tmp / "otopd_snapshot.json").write_text(
                json.dumps(_jsonable(metadata), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        # READY is the publication barrier.  Readers must never accept a
        # directory before this marker exists.
        (tmp / "READY").write_text("ok\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing adapter snapshot: {target}")
        os.replace(str(tmp), str(target))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def blend_lora_snapshots(
    previous_dir: str | os.PathLike[str] | None,
    current_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    decay: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create ``decay*previous + (1-decay)*current`` without a base model.

    If ``previous_dir`` is ``None`` the current adapter initializes the EMA.
    Configurations and tensor shapes must match exactly; silently dropping a
    LoRA tensor would invalidate tokenwise pairing, so mismatches fail closed.
    """

    import torch
    from safetensors.torch import load_file

    if not (0.0 <= float(decay) < 1.0):
        raise ValueError("ema decay must satisfy 0 <= decay < 1")
    current = validate_adapter_dir(current_dir)
    previous = validate_adapter_dir(previous_dir) if previous_dir else None
    current_cfg = json.loads((current / "adapter_config.json").read_text(encoding="utf-8"))
    current_state = load_file(str(current / "adapter_model.safetensors"), device="cpu")
    if not current_state:
        raise ValueError("current LoRA adapter is empty")

    if previous is None:
        blended = {key: value.detach().cpu() for key, value in current_state.items()}
    else:
        previous_cfg = json.loads((previous / "adapter_config.json").read_text(encoding="utf-8"))
        # Compare only semantic PEFT fields; base_model_name_or_path may be
        # absolute on one worker and relative on another.
        for key in ("peft_type", "task_type", "r", "lora_alpha", "target_modules", "target_parameters"):
            if _jsonable(previous_cfg.get(key)) != _jsonable(current_cfg.get(key)):
                raise ValueError(f"LoRA config mismatch for {key!r}")
        previous_state = load_file(str(previous / "adapter_model.safetensors"), device="cpu")
        if set(previous_state) != set(current_state):
            raise ValueError("EMA adapters have different tensor key sets")
        blended = {}
        for key in sorted(current_state):
            old = previous_state[key]
            new = current_state[key]
            if tuple(old.shape) != tuple(new.shape):
                raise ValueError(f"EMA tensor shape mismatch for {key}")
            # Accumulate in fp32, then retain the current adapter dtype.
            value = float(decay) * old.float() + (1.0 - float(decay)) * new.float()
            blended[key] = value.to(dtype=new.dtype)

    meta = dict(metadata or {})
    meta.update(
        {
            "kind": "otopsd_ema_teacher",
            "ema_decay": float(decay),
            "source_student": str(current),
            "source_teacher": str(previous) if previous else None,
        }
    )
    return _write_ready_snapshot(Path(output_dir), state=blended, config=current_cfg, metadata=meta)


__all__ = [
    "adapter_sha256",
    "blend_lora_snapshots",
    "normalise_lora_key",
    "validate_adapter_dir",
]
