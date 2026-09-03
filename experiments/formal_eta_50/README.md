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
aggregate audit JSON files for the reproducibility checks. The original
launcher logs were retained on the training machine but are not part of the
public GitHub bundle; the public `sanitized_trace.md` records the launch,
progress, exit, and final-metric markers without environment-specific noise.
The public HF step-50 adapters and their revision/hash manifest are recorded
in `metrics_summary.json`: revision
[`389104ba865e01a93f845397f8612f168f6c70c5`](https://huggingface.co/Andynsn/longvideoagent-opsd-qwen2.5-3b-lora/commit/389104ba865e01a93f845397f8612f168f6c70c5),
student SHA256 `1fbbab3290b1b42f01c6d2ebdb1babd87650ca59310d259780d97155bb94f869`,
teacher SHA256 `8218f4b99eaa3abd0c7764e99e1203116e751cf71f785539228ae39ded1cd03b`.
