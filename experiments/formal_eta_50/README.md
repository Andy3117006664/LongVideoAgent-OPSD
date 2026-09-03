# Formal ETA/EMA pilot (50 steps)

This directory contains aggregate diagnostics for the completed matched
baseline and OT-OPSD pilot.  Paths in the logs and manifests are replaced by
`$LVA_ROOT`; the raw rollout and validation JSONL files are intentionally not
included.

Key settings: Qwen2.5-3B-Instruct, 500 train / 300 validation examples,
`rollout.n=4`, two agent turns, alpha 0.1, matched-mask null span,
next-action target, EMA decay 0.99, and 50 optimizer steps.  The scorer and
trainer communicated over loopback only; no remote vision API was used.

See `metrics_summary.json` for the compact machine-readable result and the
two launcher logs for the full (sanitized) configuration and progress trace.
