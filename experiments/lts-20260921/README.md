# LTS results snapshot — 2026-09-21

The latest completed LVBench evaluation scored **57/300 (19.0%) for A1** and
**114/300 (38.0%) for A2**. Both use the same v5 step100 checkpoint. A2 adds
normalized temporal hints; both arms use the same lowercase-answer guard.
This run made **zero training updates**.

| Measurement | Result |
|---|---:|
| LVBench A1, no time hint | 57/300, 19.0% |
| LVBench A2, normalized seconds | 114/300, 38.0% |
| Paired A2−A1 | +19.0 percentage points |
| Video-cluster bootstrap 95% interval | +12.33 to +25.96 points |
| Paired wins / losses / ties | 85 / 28 / 187 |
| Latest completed training rerun, native development evaluation | 60/300, 20.0% at step150 |

The latest training rerun is `opsd16_topk_ref_coef010_confirmation_20260915_v1`.
Despite its name, the untouched confirmation evaluation has **not started**.
The earlier coefficient-0.10 development run scored 70/300; that number does
not describe the later rerun. Training development scores and LVBench scores
use different evaluations and must not be compared as a learning curve.

LVBench used 300 questions from 102 videos, seed 20260909, greedy n=1,
three assistant turns, 1024 response tokens, 256 observation tokens, and
the original 16-rank FSDP restore. Padding produced 304 execution traces
per arm; all reported accuracy and cost metrics use the 300 retained rows.
A1/A2 used 279/286 successful Inspector HTTP calls and 1160/2174 verified
returned frames. Missing and invalid answers remain incorrect.

This is a repeatedly used development set with one seed. The interval
describes this paired result and does not remove tuning-selection bias.
Inspector receives no answer labels or direct temporal annotations.

## Files and release boundary

- [metrics_summary.json](metrics_summary.json): aggregate results and training lineage.
- [evaluation_config.json](evaluation_config.json): decoding, prompt and input identities.
- [verification.json](verification.json): 2,109 local archived files rehashed, with paired counts recomputed.
- [checkpoint_manifest.json](checkpoint_manifest.json): evaluated native-checkpoint inventory.
- [SHA256SUMS](SHA256SUMS): hashes of this publication bundle.

**New checkpoints are not uploaded yet.** They remain on LTS pending an
authenticated LES connection. The companion HF repository is
[Andynsn/longvideoagent-opsd-qwen2.5-3b-lora](https://huggingface.co/Andynsn/longvideoagent-opsd-qwen2.5-3b-lora).
Its earlier step50/smoke2 adapters are not substitutes for these checkpoints.
The native step100 checkpoint needs a verified export before it can be called
a portable PEFT adapter. No base-model weights or raw evaluation dumps are
included here. This result snapshot does not update the older pilot trainer
in the repository or claim that it already reproduces the LTS runtime.
