# LV-OT-OPSD release

This repository is a reproducibility bundle for the LongVideoAgent
observation-transition self-distillation prototype.  It contains the patched
agent/trainer sources, launchers, cache-building utilities, and aggregate
experiment diagnostics.

## What is implemented

The current release implements **OT-OPSD** (observation-transition on-policy
self-distillation): the student is the actor being trained, while the scorer
keeps a lagged/EMA LoRA snapshot as teacher.  At each transition it measures
the change in teacher--student likelihood gap before and after an observation
and applies the resulting signal to the next-action span.

This is the validated transition/EMA core.  It is **not** the proposed
recursive-belief AgentOPSD or StepOPSD implementation, and the released run
uses a subtitle/local-cache observation path.  The privileged visual branch
is disabled in the reported experiment (`OT_ONLINE_PRIVILEGED=false`).

## Reproduce the formal pilot

The original run used Qwen2.5-3B-Instruct, 500 training examples, 300
validation examples, four training GPUs, `rollout.n=4`, two agent turns, and
50 optimizer steps.  The local scorer ran on a separate GPU.  A cleaned
launcher is in `tools/run_lva_opsd.sh`; set `LVA_ROOT`, model/data/cache paths,
and the `OT_OPSD_*` variables before launching.

The public bundle deliberately omits raw videos, frames, benchmark dumps,
base-model weights, virtual environments, and raw rollout JSONL files.  Those
artifacts can contain licensed or identifying content and are not needed to
verify the aggregate claims.

## Formal pilot result

The paired pilot had 800 common training rollouts and 300 common validation
examples.  Strict training accuracy was 0.14875 for baseline and 0.15250 for
OT-OPSD (delta +0.00375; McNemar two-sided p=0.869).  At validation step 50,
strict accuracy was 0.19000 versus 0.19667 (delta +0.00667; p=0.860); the
native answer accuracy was 0.19333 for both runs.  The scorer produced 276
finite observation-transition records, applied 29,982 target tokens, and
reported successful EMA refreshes on all 50 updates.

These are pilot-scale measurements, not evidence of a statistically reliable
improvement.  See `experiments/formal_eta_50/` for the aggregate audits,
machine-readable metrics, and the public `sanitized_trace.md`. The full
launcher logs remain local to the training machine; the public trace records
the relevant configuration and final status without raw rollout content.

## Checkpoint

The corresponding step-50 student and EMA-teacher LoRA adapters are published
in the companion Hugging Face repository:

`https://huggingface.co/Andynsn/longvideoagent-opsd-qwen2.5-3b-lora`

They must be loaded on top of the upstream Qwen2.5-3B-Instruct base model;
the base weights are not redistributed here.

## Attribution and license

The source tree is based on the public LongVideoAgent `newversion` snapshot
and the verl stack.  Please consult the upstream project and dependency
licenses before redistributing or deploying this bundle.  The upstream
snapshot did not include an explicit LICENSE file in the captured tree; this
repository therefore makes no blanket license claim for third-party files.
The OT-OPSD additions and experiment notes are provided for research
reproducibility by the repository owner.
