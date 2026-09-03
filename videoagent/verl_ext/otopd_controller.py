"""Driver-side lagged-actor/EMA controller for formal OT-OPSD.

The controller is intentionally tiny: FSDP workers export only their
trainable LoRA tensors, while a reserved-GPU scorer owns the frozen base
model(s).  A refresh is committed only after the scorer acknowledges both
student and teacher snapshots, so a failed reload cannot silently advance the
policy-version metadata.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


class LaggedActorController:
    """Coordinate actor LoRA snapshots and scorer reloads on the driver."""

    def __init__(
        self,
        *,
        scorer_url: str,
        snapshot_root: str,
        base_model: str,
        refresh_steps: int = 1,
        ema_decay: float | None = None,
        timeout: float = 120.0,
        mode: str = "lagged_actor",
    ) -> None:
        self.scorer_url = str(scorer_url or "").rstrip("/")
        self.snapshot_root = Path(snapshot_root).expanduser().resolve()
        self.base_model = str(base_model or "")
        self.refresh_steps = max(1, int(refresh_steps))
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        if self.ema_decay is not None and not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        self.timeout = max(1.0, float(timeout))
        self.mode = str(mode or "lagged_actor")
        self.previous_student: str | None = None
        self.previous_teacher: str | None = None
        self.previous_step: int | None = None
        self.last_status = "init"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self._restore_state()

    @property
    def enabled(self) -> bool:
        return bool(self.scorer_url)

    def _state_path(self) -> Path:
        return self.snapshot_root / "latest.json"

    def _restore_state(self) -> None:
        path = self._state_path()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            student = state.get("student_adapter")
            teacher = state.get("teacher_adapter")
            if student and Path(student).is_dir() and (Path(student) / "READY").is_file():
                self.previous_student = str(student)
            if teacher and Path(teacher).is_dir() and (Path(teacher) / "READY").is_file():
                self.previous_teacher = str(teacher)
            if state.get("student_step") is not None:
                self.previous_step = int(state["student_step"])
        except Exception:
            # A stale/incomplete state file must never block a fresh run.
            self.previous_student = None
            self.previous_teacher = None
            self.previous_step = None

    def _publish_state(self, state: dict[str, Any]) -> None:
        target = self._state_path()
        fd, tmp_name = tempfile.mkstemp(prefix=".latest.", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _snapshot_metadata(path: str | Path) -> dict[str, Any]:
        """Read optional publication metadata without trusting its strings."""

        try:
            value = json.loads((Path(path) / "otopd_snapshot.json").read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _ready_snapshot(cls, path: str | Path, *, expected_step: int | None = None) -> bool:
        candidate = Path(path)
        if not (
            candidate.is_dir()
            and (candidate / "READY").is_file()
            and (candidate / "adapter_config.json").is_file()
            and (
                (candidate / "adapter_model.safetensors").is_file()
                or (candidate / "adapter_model.bin").is_file()
            )
        ):
            return False
        if expected_step is None:
            return True
        metadata = cls._snapshot_metadata(candidate)
        try:
            return int(metadata.get("global_step", metadata.get("student_step"))) == int(expected_step)
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _rpc_result(value: Any) -> dict[str, Any]:
        """Extract rank-zero result from ONE_TO_ALL worker output."""

        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    try:
                        rank_zero = int(item.get("rank", 1)) == 0
                    except (TypeError, ValueError, OverflowError):
                        rank_zero = False
                    if item.get("published") or rank_zero:
                        return item
            for item in value:
                if isinstance(item, dict):
                    return item
        raise RuntimeError(f"save_lora_adapter returned invalid result: {value!r}")

    def _reload_scorer(
        self,
        *,
        student_adapter: str,
        teacher_adapter: str,
        student_step: int,
        teacher_step: int,
    ) -> dict[str, Any]:
        payload = {
            "base_model": self.base_model,
            "student_adapter": student_adapter,
            "teacher_adapter": teacher_adapter,
            "student_step": int(student_step),
            "teacher_step": int(teacher_step),
            "snapshot_mode": self.mode,
            "ema_decay": self.ema_decay,
        }
        request = urllib.request.Request(
            self.scorer_url + "/reload",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(f"scorer rejected OT-OPSD reload: {result!r}")
        # A 200 response is not enough: a delayed HTTP retry must not make the
        # trainer believe that an older pair is active.  Require the scorer to
        # acknowledge this exact monotone version and adapter paths.
        try:
            acknowledged_version = result.get("version")
            if acknowledged_version is None:
                raise RuntimeError("scorer response omitted snapshot version")
            if int(acknowledged_version) != int(student_step):
                raise RuntimeError(
                    f"scorer acknowledged version {acknowledged_version!r}, expected {student_step}"
                )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"scorer returned an invalid snapshot version: {result!r}") from exc
        for key, expected in (("student_adapter", student_adapter), ("teacher_adapter", teacher_adapter)):
            acknowledged = result.get(key)
            if acknowledged is None:
                raise RuntimeError(f"scorer response omitted {key}")
            if str(Path(str(acknowledged)).resolve()) != str(Path(expected).resolve()):
                raise RuntimeError(
                    f"scorer acknowledged {key}={acknowledged!r}, expected {expected!r}"
                )
        return result

    def refresh(self, actor_worker_group: Any, *, global_step: int, force: bool = False) -> dict[str, Any]:
        """Export actor LoRA, build lagged/EMA teacher, and reload scorer.

        Call this after ``update_actor`` *and* after rollout weights have been
        synchronized.  The next rollout then uses the same student snapshot
        that the scorer reports in ``ot_student_step``.
        """

        step = int(global_step)
        if not self.enabled:
            self.last_status = "disabled"
            return {"otopsd/refresh_skipped": 1.0, "otopsd/status": "disabled"}
        if not force and step % self.refresh_steps != 0:
            self.last_status = "interval_skip"
            return {"otopsd/refresh_skipped": 1.0, "otopsd/status": "interval_skip"}

        step_root = self.snapshot_root / f"step_{step}"
        student_dir = step_root / "student"
        teacher_dir = step_root / "teacher"
        try:
            # A trainer can be restarted after rank zero published the
            # adapter but before the scorer ACK/state file was written.  Reuse
            # that complete directory instead of asking the worker to
            # overwrite it (the worker intentionally refuses overwrites).
            if self._ready_snapshot(student_dir, expected_step=step):
                student_result = {"ok": True, "path": str(student_dir), "reused": True}
            elif Path(student_dir).exists():
                raise RuntimeError(f"conflicting/incomplete actor snapshot directory: {student_dir}")
            else:
                rpc = actor_worker_group.save_lora_adapter(str(student_dir), step, self.base_model)
                student_result = self._rpc_result(rpc)
                if student_result.get("ok") is not True:
                    raise RuntimeError(f"actor LoRA snapshot was not published: {student_result!r}")
            if not Path(student_dir).is_dir() or not (student_dir / "READY").is_file():
                raise RuntimeError(f"actor LoRA snapshot was not published: {student_result!r}")

            if self.ema_decay is None:
                # Hard lagged actor: teacher is exactly the previously
                # committed student.  At step zero, initialize teacher=student
                # (e_obs is expected to be zero until the first update).
                teacher_source = self.previous_student or str(student_dir)
                teacher_step = self.previous_step if self.previous_step is not None else step
                if teacher_source == str(student_dir):
                    teacher_dir = student_dir
                else:
                    teacher_dir = Path(teacher_source)
            else:
                try:
                    from .otopd_adapter import blend_lora_snapshots
                except ImportError:
                    from otopd_adapter import blend_lora_snapshots

                # EMA output is also published atomically.  If a previous
                # attempt completed this exact step, it is safe to reuse it;
                # otherwise a conflicting directory is a hard failure rather
                # than an overwrite of an unknown teacher.
                if self._ready_snapshot(teacher_dir):
                    existing_meta = self._snapshot_metadata(teacher_dir)
                    source_ok = str(existing_meta.get("source_student") or "") == str(Path(student_dir).resolve())
                    try:
                        decay_ok = float(existing_meta.get("ema_decay")) == float(self.ema_decay)
                    except (TypeError, ValueError, OverflowError):
                        decay_ok = False
                    if not (source_ok and decay_ok):
                        raise RuntimeError(f"conflicting EMA teacher snapshot directory: {teacher_dir}")
                elif Path(teacher_dir).exists():
                    raise RuntimeError(f"incomplete EMA teacher snapshot directory: {teacher_dir}")
                else:
                    blend_lora_snapshots(
                        self.previous_teacher,
                        str(student_dir),
                        str(teacher_dir),
                        decay=self.ema_decay,
                        metadata={"student_step": step, "teacher_step": self.previous_step},
                    )
                teacher_step = step

            reload_result = self._reload_scorer(
                student_adapter=str(student_dir),
                teacher_adapter=str(teacher_dir),
                student_step=step,
                teacher_step=int(teacher_step),
            )
            self.previous_student = str(student_dir)
            self.previous_teacher = str(teacher_dir)
            self.previous_step = step
            self.last_status = "reloaded"
            self._publish_state(
                {
                    "student_adapter": self.previous_student,
                    "teacher_adapter": self.previous_teacher,
                    "student_step": step,
                    "teacher_step": int(teacher_step),
                    "mode": self.mode,
                    "ema_decay": self.ema_decay,
                    "scorer": reload_result,
                }
            )
            return {
                "otopsd/refresh_ok": 1.0,
                "otopsd/refresh_skipped": 0.0,
                "otopsd/student_step": float(step),
                "otopsd/teacher_step": float(teacher_step),
                "otopsd/status": "reloaded",
            }
        except Exception as exc:
            self.last_status = f"error:{type(exc).__name__}"
            return {
                "otopsd/refresh_ok": 0.0,
                "otopsd/refresh_skipped": 0.0,
                "otopsd/status": self.last_status,
                "otopsd/error": str(exc)[:240],
            }


__all__ = ["LaggedActorController"]
