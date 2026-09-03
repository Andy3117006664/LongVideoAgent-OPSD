# Formal pilot — public execution trace

This is a compact, manually filtered trace of the two completed 50-step
jobs. It contains launch configuration, progress/exit markers, and aggregate
metrics only; the original verbose launcher output stays on the training
machine because it includes environment-specific diagnostics.

## Matched baseline

```text
[config] mode=baseline alpha=0 gpus=1,2,3,6
[config] train_examples=500 validation_examples=300 steps=50 batch=4 rollout_n=4
[config] observation=subtitle_local_cache grounding=local_bm25_cache remote_api=false
[progress] 50/50 optimizer steps; elapsed=1:01:56
[metric] train_strict_accuracy=0.14875 n=800
[metric] val50_strict_accuracy=0.19000 n=300
[metric] val50_native_accuracy=0.1933333333
[done] status=0
```

## OT-OPSD transition/EMA

```text
[config] mode=ot alpha=0.1 gpus=1,2,3,6
[config] train_examples=500 validation_examples=300 steps=50 batch=4 rollout_n=4
[config] observation=subtitle_local_cache grounding=local_bm25_cache remote_api=false
[otopsd] static_ot_cache=empty; signal_source=local_loopback_paired_scorer
[otopsd] ema_decay=0.99 refresh_interval=1 scorer=loopback
[otopsd] refresh_ok=50/50 refresh_skipped=0
[ot] finite_records=276 applied_records=276 applied_tokens=29982
[progress] 50/50 optimizer steps; elapsed=1:06:28
[metric] train_strict_accuracy=0.15250 n=800
[metric] val50_strict_accuracy=0.1966666667 n=300
[metric] val50_native_accuracy=0.1933333333
[done] status=0
```

The exact paired tests, cache-leak audit, HF revision, and adapter hashes are
in `metrics_summary.json` and the other JSON files in this directory.
