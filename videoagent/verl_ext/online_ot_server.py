"""Small reserved-GPU HTTP service for LongVideoAgent online OT scoring.

The rollout workers can set ``ot_online_url`` to this service instead of
loading a teacher/student pair in every Ray process.  The service keeps one
process-local :class:`OnlinePairedScorer` and exposes ``POST /score`` with the
same JSON contract consumed by ``agent_loop.py``.

Example::

    python online_ot_server.py \
      --student-model /models/qwen2.5-3b \
      --teacher-model /models/qwen2.5-7b \
      --device cuda:0 --port 8765

Then configure the rollout workers with
``ot_online_url=http://127.0.0.1:8765``.  Keep the service on a trusted local
interface; authentication and multi-tenant isolation are intentionally out of
scope for this experiment helper.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any


logger = logging.getLogger("online_ot_server")


def create_app(
    *,
    student_model: str,
    teacher_model: str,
    device: str = "cuda:0",
    dtype: str = "auto",
    trust_remote_code: bool = True,
):
    """Build a FastAPI app and eagerly load one paired scorer."""

    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover - exercised only on launch
        raise RuntimeError("online_ot_server requires fastapi and uvicorn") from exc

    try:
        try:
            from .online_ot_scorer import OnlinePairedScorer
        except ImportError:
            from online_ot_scorer import OnlinePairedScorer

        scorer = OnlinePairedScorer.get(
            student_model=student_model,
            teacher_model=teacher_model,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to load paired scorer: {type(exc).__name__}: {exc}") from exc

    app = FastAPI(title="LongVideoAgent online OT scorer")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "status": "ready"}

    @app.post("/score")
    def score(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        prompt_ids = payload.get("prompt_ids")
        response_ids = payload.get("response_ids")
        response_mask = payload.get("response_mask")
        records = payload.get("records")
        if not all(isinstance(value, list) for value in (prompt_ids, response_ids, response_mask, records)):
            raise HTTPException(status_code=400, detail="prompt/response/mask/records must be lists")
        if len(response_ids) != len(response_mask):
            raise HTTPException(status_code=400, detail="response_ids and response_mask lengths differ")
        if not all(isinstance(record, dict) for record in records):
            raise HTTPException(status_code=400, detail="records must contain JSON objects")
        try:
            hits, misses, status = scorer.score_records(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                records=records,
                reduction=str(payload.get("reduction", "mean")),
                batch_size=max(1, int(payload.get("batch_size", 2))),
                max_records=max(0, int(payload.get("max_records", 0))),
            )
        except Exception as exc:  # defensive boundary; scorer is fail-closed
            logger.exception("paired scoring request failed")
            return {
                "ok": True,
                "status": f"error:{type(exc).__name__}",
                "hits": 0,
                "misses": len(records),
                "records": records,
            }
        return {
            "ok": True,
            "status": str(status),
            "hits": int(hits),
            "misses": int(max(0, len(records) - hits)),
            "records": records,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - exercised only on launch
        raise RuntimeError("online_ot_server requires uvicorn") from exc
    app = create_app(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
