# LongVideoAgent-OPSD 协作 Agent 操作指南

本文给负责维护代码、在其他机器复现实验、上传结果的 agent 使用。先读完本文件，再执行任何训练或公开上传。

## 公开仓库

- GitHub（代码、启动器、脱敏日志和指标）：<https://github.com/Andy3117006664/LongVideoAgent-OPSD>
- Hugging Face（LoRA checkpoint）：<https://huggingface.co/Andynsn/longvideoagent-opsd-qwen2.5-3b-lora>
- 上游 LongVideoAgent：<https://github.com/longvideoagent/LongVideoAgent>

GitHub 与 HF 都是公开仓库。任何 token、`.env`、私钥、机器凭据、原始视频/帧、未经许可的数据集、完整基础模型权重都禁止提交。

## 先弄清楚方法边界

当前代码是 **LV-OT-OPSD transition/EMA core**：训练中的 actor 是 student，scorer 保存滞后/EMA LoRA 作为 teacher，并以

`e_obs = D(h+) - D(h-)`, `D = log pi_teacher - log pi_student`

计算 observation-transition 信号，再作用到 next-action span。它是严格的同一 actor 的 EMA/lagged self-distillation，而不是外部冻结 LongVideoAgent checkpoint 的 OPD。

当前发布的正式 run 使用 subtitle/local-cache observation，`OT_ONLINE_PRIVILEGED=false`；因此不要把它描述成已经完成的递归 belief AgentOPSD、StepOPSD 或 privileged visual OPSD。要做这些扩展，必须另开 feature branch、定义 answer-free privileged cache，并增加对应审计。

## 已完成结果与作业记录

| run | 状态/配置 | 主要结果 |
|---|---|---|
| `lva_baseline_all4198_50_s20260903` | baseline，50 steps，500 train/300 val，4 张 RTX 6000D | train strict 0.14875 (n=800)，val@50 strict 0.19000 |
| `matched_500train_300val_formal_eta_50_s20260903` | OT-OPSD，alpha=0.1，EMA decay=0.99，`rollout.n=4`，50 steps，4 张训练卡 + 1 张 scorer 卡 | train strict 0.15250 (n=800)，val@50 strict 0.19667，native val accuracy 0.19333；paired McNemar p=0.869/0.860；276 个 finite `e_obs`，29,982 个 applied tokens，50/50 refresh 成功 |
| `lva_opsd_repeat8_50_s20260903` | 已因验证窗口过长而停止（未进入训练更新）；不作为结果发布 | 仅保留作业记录，不要当作成功 run |
| `lva_opsd_repeat8_canary10_s20260903` | 已因串行验证窗口过长而停止（约 3/32，未进入训练更新） | 仅用于定位验证吞吐瓶颈，不作为结果 |
| `lva_opsd_repeat8_smoke2_s20260903` | **已完成**；seed 20260906，2 steps，16 train / 2 val，8 张训练卡，scorer 与 0 号卡共存 | exit 0；step 0/1/2 reload 成功；2/2 refresh 成功；step-2 student/teacher 已公开到 HF `smoke2/step_2/`（revision `f6fc82ca0fc384cf405dcea3e1c6a87be2a9ccfa`）。OT cache 为空且两步均 `skipped_no_effect=1`，所以只证明 plumbing，不证明方法效果 |

正式 run 的完整 aggregate audit 在 `experiments/formal_eta_50/`。8 卡 smoke
的配置、脱敏日志与机器可读摘要在 `experiments/repeat8_smoke2/`；其 checkpoint
已位于公开 HF 的 `smoke2/step_2/`，不要覆盖正式 checkpoint 的 `step_50/`。
HF revision 与 adapter hash 已写入摘要。pilot 的
小幅提升没有统计显著性，后续报告必须保留这个结论。

## 分支与提交规则

1. `main` 只放可复现、已审计的代码和文档；不要直接在 `main` 上试验，也不要 force-push。
2. 新方法用 `feature/<topic>`，新实验/参数用 `exp/<date>-<name>`（例如 `exp/20260903-opsd-repeat8`）。本地 Codex 工作分支可使用 `codex/<topic>`。
3. 一条 branch 同一时间只指定一个维护者/agent。不同机器不要共同向同一实验 branch 推送；若要接手，先让原维护者推完并在 PR/issue 留下最新 commit SHA，再从该 SHA 新建接力 branch。
4. 开工前执行 `git fetch origin`，从最新 `origin/main` 建 branch；完成后 push branch、开 PR，由另一位维护者检查指标、许可证和 secret scan，再 squash/merge 到 `main`。
5. 一个实验一个唯一 `RUN_NAME`/输出目录；禁止覆盖已有 `outputs/*`。
6. 合并前必须提交：配置快照、seed、数据/缓存 hash、模型基座名称、GPU 数、step 数、聚合指标、失败原因（若失败）以及脱敏后的日志。
7. PR 只包含源码/脚本/文档/小型 aggregate JSON；checkpoint 放 HF，不把大文件塞进 GitHub。
8. 合并前运行 secret scan，并检查 `git diff --stat`；发现凭据或原始数据立即停止上传。
9. 稳定版本打 tag（例如 `v0.1-opsd-transition`），不要重写公开历史。

## 在新机器上接手

```bash
git clone https://github.com/Andy3117006664/LongVideoAgent-OPSD.git
cd LongVideoAgent-OPSD
git switch main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_for_newversion.txt
```

准备本地（不要提交）目录：

```text
$LVA_ROOT/
  repo/                       # 本仓库 checkout
  models/Qwen2.5-3B-Instruct/ # 由使用者按上游条款自行准备
  data/
    LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json
    frames/none/              # subtitle-only smoke 可为空目录
    frames/none.json          # subtitle-only smoke 的空 bbox JSON
    parquet_5choice_500/train.parquet
    parquet_5choice_val_300/train.parquet
    local_grounding_bm25_v1.json
    subtitle_observation_proxy_all4198_v3.json
    cache/<nonempty-ot-records>.json
  env/                        # 运行环境（不上传）
```

`SUBS`（clip-level subtitle JSON）是 launcher 的硬性必需项；`FRAME_DIR` 与
`BBOX_JSON` 也会写入配置。上面的 `frames/none*` 只适用于 subtitle-only
smoke，启用真实 visual-query 时必须换成有授权的帧目录和 bbox 文件。

训练只使用本地 cache；默认不会请求远程视觉 API。先把路径替换成当前机器的绝对路径，再执行 `DRY_RUN=1` 预检。

## 开启一个新的 LV-OT-OPSD 实验

1. 复制一个新的 `experiments/<run>/` 配置和 README，写明 seed、数据子集、cache hash、基座模型和假设。
2. 选择空闲 GPU，并保证 `TRAIN_BSZ * ROLLOUT_N` 及 `PPO_MINI * ROLLOUT_N` 可被训练 world size 整除。
3. 启动 loopback scorer（scorer 可与训练的 0 号卡共存，但先检查显存）：

```bash
BASE=/path/to/lva-root
OUT="$BASE/outputs/<unique-run>"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 nohup "$BASE/env/bin/python" \
  "$BASE/repo/videoagent/verl_ext/online_opsd_server.py" \
  --base-model "$BASE/models/Qwen2.5-3B-Instruct" \
  --device cuda:0 --host 127.0.0.1 --port 8765 \
  > "$OUT/scorer.log" 2>&1 < /dev/null &
```

4. 运行 `tools/run_lva_opsd.sh`。下面的八卡命令是 **smoke/plumbing 模板**（它故意使用空 OT cache，scorer 与 0 号卡共存）；按机器实际路径改写：

> **不要用这段空 cache 配置声称复现 formal OT 效果。** 正式实验必须提供
> 可审计的非零 transition 信号：要么使用与 rollout 同轨迹生成的非空 OT
> replay cache（删除 `ALLOW_EMPTY_OT_CACHE=1` 并记录 cache SHA256），要么
> 明确启用并记录本地 loopback paired scorer。当前 formal pilot 采用后者，
> 产生了 276 条 online records；本段空 cache 命令本身仍只用于检查 8 卡启动、
> EMA snapshot、scorer reload 和退出流程。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LVA_ROOT="$BASE" MODE=ot OUT="$OUT" \
TRAIN="$BASE/data/parquet_5choice_500/train.parquet" \
VAL="$BASE/data/parquet_5choice_val_300/train.parquet" \
SUBS="$BASE/data/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json" \
FRAME_DIR="$BASE/data/frames/none" BBOX_JSON="$BASE/data/frames/none.json" \
OBSERVATION_CACHE="$BASE/data/subtitle_observation_proxy_all4198_v3.json" \
GROUNDING_CACHE="$BASE/data/local_grounding_bm25_v1.json" \
OT_CACHE="$BASE/data/cache/empty_ot.json" ALLOW_EMPTY_OT_CACHE=1 \
ALLOW_GENERIC_OBS_CACHE=true TOTAL_STEPS=50 TRAIN_BSZ=4 ROLLOUT_N=4 \
PPO_MINI=4 VAL_BSZ=1 TRAIN_MAX_SAMPLES=500 VAL_MAX_SAMPLES=300 \
MAX_TURNS=2 AGENT_WORKERS=1 SEED=<new-seed> \
OT_OPD_NULL_MODE=matched_mask OT_OPD_ALPHA=0.1 \
OT_OPD_SIGNAL_KEY=e_obs OT_OPD_NEGATE_SIGNAL=1 \
OT_OPD_TARGET_SPAN=next_action OT_OPD_MODE=multiplier \
OT_OPD_MULTIPLIER_BOUND=0.5 OT_OPD_NORMALIZE=true OT_OPD_CLIP=2.0 \
OT_ONLINE_SCORE=true OT_ONLINE_URL=http://127.0.0.1:8765 \
OT_ONLINE_STANDARD_GAP=true OT_ONLINE_PRIVILEGED=false \
OT_ONLINE_TIMEOUT=300 OT_ONLINE_BATCH_SIZE=4 \
OT_OPSD_LAGGED=1 OT_OPSD_URL=http://127.0.0.1:8765 \
OT_OPSD_BASE_MODEL="$BASE/models/Qwen2.5-3B-Instruct" \
OT_OPSD_SNAPSHOT_DIR="$OUT/otopsd_lora" OT_OPSD_REFRESH_STEPS=1 \
OT_OPSD_EMA_DECAY=0.99 OT_OPSD_REQUIRE=1 \
DRY_RUN=1 bash tools/run_lva_opsd.sh
```

预检通过后，把 `DRY_RUN=1` 改成 `DRY_RUN=0`，并将 stdout/stderr 保存在该 run 目录。训练结束必须检查 `status=0`、`otopsd/refresh_ok`、snapshot `READY` 文件和 aggregate audit；scorer 停止前先保存其日志。正式 run 还必须确认 `ot_opd/applied_records>0` 且 `ot_opd/skipped_no_effect` 不占满所有更新。

当前 launcher 的 `TOTAL_STEPS` 上限是 50。要超过 50 steps，先在独立 branch 实现并验证 checkpoint/resume 语义，再开新 run；不要简单删除上限或覆盖旧目录。现有 LoRA snapshot 不是完整 optimizer checkpoint，默认不能无缝续训，只能以新 seed/新输出目录重跑，除非 resume 功能已被单独审计。

## 从另一台机器上传什么

### 上传 GitHub

必须上传：

- 修改后的 Python/Shell 源码、`tools/` launcher、必要的 cache builder；
- `experiments/<run>/README.md`、`metrics_summary.json`、配置快照和 SHA256 manifest；
- 脱敏后的 launcher/scorer 日志（相对路径、无 token、无原始 prompt/answer）；
- 失败 run 的短错误摘要，便于协作者复现。

禁止上传：

- `env/`、`.venv/`、`__pycache__/`、pid/socket、模型基础权重；
- 原始视频、帧 tar、字幕/benchmark dump、完整 rollout/validation JSONL（除非另有数据许可并已审核）；
- `.env`、shell history、HF/GitHub token、任何 `Authorization` header。

### 上传 Hugging Face

默认只上传最终 step 的两个 adapter 目录。正式 run 保留在 `step_50/`；
其他实验必须使用独立前缀，例如 `runs/<RUN_NAME>/step_<N>/`，不得覆盖已有路径：

```text
step_50/student/{adapter_config.json, adapter_model.safetensors, ...}
step_50/teacher/{adapter_config.json, adapter_model.safetensors, ...}
```

同时上传 `README.md`、训练配置、base model 指针、step/seed/EMA 信息和 SHA256。中间 0..50 快照约 5.8 GB，除非明确需要做轨迹分析，否则不上传。HF repo 必须保持 public，但 token 只能通过本机登录状态或安全环境变量传入，不能写入命令日志。

建议用 HF CLI/API 的 `upload_folder`，上传前确认目标是 `Andynsn/longvideoagent-opsd-qwen2.5-3b-lora`，并在本地列出待上传文件清单。上传后必须重新列远端文件、确认 repo 为 public、下载或读取 manifest，并比对两个 safetensors 的 SHA256；把 HF commit/revision 写回实验 README。

### 停机前的最短闭环

1. 等 launcher 明确写出 `status=0`，确认 final step 的 student/teacher 都有 `READY`。
2. 计算两个 adapter 的 SHA256，先上传 HF final step 和 manifest；确认远端文件大小/hash 后，再处理 GitHub 文档。
3. 从日志只保留配置、step 指标、refresh 和退出行，去掉绝对路径、token、原始 prompt/answer；把 `README.md`、`metrics_summary.json` 和脱敏日志推到实验 branch。
4. 开 PR 或在交接记录中写清 Git commit、HF revision、未上传的本地数据、失败/中止作业和下一步。
5. 只停止本 run 的 scorer/worker PID，重新检查 GPU、端口和进程；不要用宽泛的 `pkill`。

## 合作者/Agent 的工作协议

- 接手先读本指南、`OPSD_RELEASE.md` 和对应 experiment README；先报告 `git status`、当前 branch、GPU/磁盘和目标 run。
- 任何参数改变都写入新的 config/README，并说明为何改变；不把“transition core”说成递归 belief。
- 训练中的异常先保留日志和进程信息，再做最小范围停止；不要 `kill -9` 不属于本 run 的进程。
- 完成后给出：run 路径、exit status、step 数、refresh 成功率、finite/applied record 数、train/val 指标、artifact hashes、GitHub commit 和 HF revision。
- 发现数据授权、凭据、显存或代码来源不清时，暂停公开上传并请求人工确认。

## 下一步研究计划

1. 先在 `feature/parallel-validation` 解决 8 卡作业仍被单样本串行验证拖慢的问题，并用固定 16-train/32-val canary 做吞吐与一致性审计；不要把 smoke 的 58 秒更新耗时外推为完整实验时长。
2. 从最新 `main` 开 `exp/<date>-transition-multiseed`，在非空 online OT 信号、完整 300-val 和至少 3 个 seed 下复现 matched baseline/OT-OPSD；报告 paired bootstrap/McNemar、长度分桶和工具调用分桶。
3. 在 `feature/privileged-visual-opsd` 实现 `OT_ONLINE_PRIVILEGED=true` 的 answer-free visual cache，单独比较 `e_obs`、`e_priv` 和无 privileged 分支，并做答案泄漏审计。
4. 在独立 `feature/recursive-belief-stepopsd` 实现递归 belief/StepOPSD：显式 belief state、step-level credit、长程回溯；先写单测与最小 canary，再开正式实验。
5. 只有在多个 seed、完整验证集和泄漏审计通过后，才把结果写成方法收益；当前 50-step pilot 只能报告“未观察到统计显著提升”，8 卡 smoke 只能报告“训练/EMA refresh 链路通过”。
