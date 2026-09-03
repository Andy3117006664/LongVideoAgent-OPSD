# 8-GPU OT-OPSD smoke (`lva_opsd_repeat8_smoke2_s20260903`)

This bounded plumbing run completed on 2026-09-03 with eight RTX 6000D
training workers. It verifies that the actor/EMA-teacher loop can refresh the
scorer and finish a short PPO run. It is **not** a benchmark result and must
not be compared with the 50-step formal pilot.

## Configuration

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- World size: 8 (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`)
- Training examples: 16; validation examples: 2
- Optimizer steps: 2; train batch: 4; `rollout.n=4`
- `max_turns=1`, `max_obs_length=128`, `max_prompt_length=2048`,
  `max_response_length=128`
- PPO mini-batch: 4; learning rate: `5e-6`; seed: `20260906`
- Observation/grounding: local subtitle and BM25 caches; remote APIs disabled
- EMA teacher: decay 0.99, refresh every step, `snapshot_mode=ema_actor`
- OT cache: empty, so this run tests OPSD snapshot/refresh plumbing but not a
  non-zero observation-transition effect

## Outcome

- Exit status `0`; training progress `2/2`
- Scorer reloaded at steps 0, 1, and 2; update steps 1 and 2 both report
  `refresh_ok=1`, `refresh_skipped=0`
- Final snapshot: step 2 student and EMA-teacher adapters, `world_size=8`
- The directory is `step_2`; the EMA teacher metadata records its source
  snapshot as step 1 (`teacher_step=1`) while the reload/controller ACK is step 2.
- Final validation accuracy: `1.0` on only two examples; no statistical meaning
- Both updates report `ot_opd/skipped_no_effect=1.0` because the OT cache was
  intentionally empty

Raw rollout/validation JSONL and local datasets are deliberately not public.
The GitHub bundle contains this aggregate description and filtered audit logs.
The step-2 adapters are public in the companion HF repository under
[`smoke2/step_2/`](https://huggingface.co/Andynsn/longvideoagent-opsd-qwen2.5-3b-lora/tree/f6fc82ca0fc384cf405dcea3e1c6a87be2a9ccfa/smoke2/step_2)
at revision `f6fc82ca0fc384cf405dcea3e1c6a87be2a9ccfa`. The formal public HF
release remains under `step_50/`.
