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

## 已完成结果与正在运行的结果

| run | 状态/配置 | 主要结果 |
|---|---|---|
| `lva_baseline_all4198_50_s20260903` | baseline，50 steps，500 train/300 val，4 张 RTX 6000D | train strict 0.14875 (n=800)，val@50 strict 0.19000 |
| `matched_500train_300val_formal_eta_50_s20260903` | OT-OPSD，alpha=0.1，EMA decay=0.99，`rollout.n=4`，50 steps，4 张训练卡 + 1 张 scorer 卡 | train strict 0.15250 (n=800)，val@50 strict 0.19667，native val accuracy 0.19333；paired McNemar p=0.869/0.860；276 个 finite `e_obs`，29,982 个 applied tokens，50/50 refresh 成功 |
| `lva_opsd_repeat8_50_s20260903` | **当前运行中**；seed 20260904，8 张卡均加入训练，scorer 与 0 号卡共存 | 结束后从远端 `outputs/lva_opsd_repeat8_50_s20260903` 读取，不要提前宣称结果 |

正式 run 的完整 aggregate audit 在 `experiments/formal_eta_50/`。pilot 的小幅提升没有统计显著性，后续报告必须保留这个结论。

## 分支与提交规则

1. `main` 只放可复现、已审计的代码和文档；不要直接在 `main` 上试验。
2. 新方法用 `feature/<topic>`，新实验/参数用 `exp/<date>-<name>`（例如 `exp/20260903-opsd-repeat8`）。本地 Codex 工作分支可使用 `codex/<topic>`。
3. 一个实验一个唯一 `RUN_NAME`/输出目录；禁止覆盖已有 `outputs/*`。
4. 合并前必须提交：配置快照、seed、数据/缓存 hash、模型基座名称、GPU 数、step 数、聚合指标、失败原因（若失败）以及脱敏后的日志。
5. PR 只包含源码/脚本/文档/小型 aggregate JSON；checkpoint 放 HF，不把大文件塞进 GitHub。
6. 合并前运行 secret scan，并检查 `git diff --stat`；发现凭据或原始数据立即停止上传。
7. 稳定版本打 tag（例如 `v0.1-opsd-transition`），不要重写公开历史。

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
  data/                       # 使用者有权使用的 LongTVQA/cache
  env/                        # 运行环境（不上传）
```

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

4. 运行 `tools/run_lva_opsd.sh`。八卡模板（scorer 与 0 号卡共存）如下；按机器实际路径改写：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LVA_ROOT="$BASE" MODE=ot OUT="$OUT" \
TRAIN="$BASE/data/parquet_5choice_500/train.parquet" \
VAL="$BASE/data/parquet_5choice_val_300/train.parquet" \
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

预检通过后，把 `DRY_RUN=1` 改成 `DRY_RUN=0`，并将 stdout/stderr 保存在该 run 目录。训练结束必须检查 `status=0`、`otopsd/refresh_ok`、snapshot `READY` 文件和 aggregate audit；scorer 停止前先保存其日志。

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

默认只上传最终 step 的两个 adapter 目录：

```text
step_50/student/{adapter_config.json, adapter_model.safetensors, ...}
step_50/teacher/{adapter_config.json, adapter_model.safetensors, ...}
```

同时上传 `README.md`、训练配置、base model 指针、step/seed/EMA 信息和 SHA256。中间 0..50 快照约 5.8 GB，除非明确需要做轨迹分析，否则不上传。HF repo 必须保持 public，但 token 只能通过本机登录状态或安全环境变量传入，不能写入命令日志。

建议用 HF CLI/API 的 `upload_folder`，上传前确认目标是 `Andynsn/longvideoagent-opsd-qwen2.5-3b-lora`，并在本地列出待上传文件清单。

## 合作者/Agent 的工作协议

- 接手先读本指南、`OPSD_RELEASE.md` 和对应 experiment README；先报告 `git status`、当前 branch、GPU/磁盘和目标 run。
- 任何参数改变都写入新的 config/README，并说明为何改变；不把“transition core”说成递归 belief。
- 训练中的异常先保留日志和进程信息，再做最小范围停止；不要 `kill -9` 不属于本 run 的进程。
- 完成后给出：run 路径、exit status、step 数、refresh 成功率、finite/applied record 数、train/val 指标、artifact hashes、GitHub commit 和 HF revision。
- 发现数据授权、凭据、显存或代码来源不清时，暂停公开上传并请求人工确认。

## 下一步研究计划

1. 先收集 `lva_opsd_repeat8_50_s20260903` 的 8 卡结果，与已有 matched baseline 做同 seed/同 cache 的 paired audit。
2. 若 transition 信号稳定，再实现 `OT_ONLINE_PRIVILEGED=true` 的 answer-free visual cache，并单独比较 `e_obs`、`e_priv` 和无 privileged 分支。
3. 再开 branch 实现递归 belief/StepOPSD（显式 belief state、step-level credit、长程回溯），补充长度分桶、工具调用类型和多 seed 统计。
4. 只有在至少多个 seed、完整验证集和通过泄漏审计后，才把结果写成方法收益；当前 pilot 只能报告“未观察到统计显著提升”。
