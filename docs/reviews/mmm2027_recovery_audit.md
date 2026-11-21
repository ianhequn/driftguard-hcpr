# MMM 2027 Post-Review 恢复与可复现性审计

审计工作区：`D:\DriftGuard-mmm-revision`，分支 `revision/mmm2027-post-review`，基线提交 `60c39112a6b580bf24cf07e57fc1f5cd1a293d9c`。

## 1. 安全恢复状态

- 原工作区 `D:\DriftGuard` 保持在 `research/round10-reuse-boundary`，存在用户未提交修改；本轮没有切换、reset、clean 或覆盖它。
- 新 worktree 已从 `60c39112` 创建，目标目录此前不存在。
- 当前审计分支只承载 post-review 文件；历史 Round 10/11 artifact 保留原样。

## 2. 资产完整性

| 项目 | 结果 | 证据与说明 |
|---|---|---|
| HCPR 核心代码 | 基本完整 | `driftguard/structured_reuse/hcpr.py`、`r11b.py`、`pipeline.py` 存在；`tests/test_hcpr_pipeline.py`、`test_round11b_variants.py` 存在 |
| Round 11B 结果 | artifact 完整 | `results/R11B_E00_preflight` 到 `R11B_E06_analysis` 均存在；E03 transfer、E04 latency、E05 quality/latency、E06 analysis 齐全 |
| Round 11A 原始结果 | worktree 不完整，备份存在 | 当前分支保留 Round 11A 汇总报告和 CSV；`D:\r11a-final-results.tgz` 含 R11A raw result 目录 |
| 论文 source/PDF | 存在 | `paper/mmm2027/MMM2027_HCPR_anonymous_revised.tex/pdf` |
| VSPW 结构 | 存在 | `D:\datasets\VSPW_480p\VSPW_480p` 含 `train.txt/val.txt/test.txt`、`data/`、label dictionary；本地 `data/` 有 3,536 个 video 目录 |
| VSPW val contract | 历史审计通过 | `results/R765_E01_domain_scan/full_domain_scan.json` 记录 val 24,392 masks、343 videos；canonical invalid 已归一到 255 |
| per-frame prediction masks | 缺失 | R11B 目录只有 aggregate summary、route、latency、stage calls；搜索不到 HCPR 的 per-frame mask 输出 |
| two-action baseline | 缺失 | 代码、脚本和结果中没有 Full↔S2-S4 controller 或对应完整曲线 |

## 3. Checkpoint provenance

论文冻结模型的预期 SHA256：

- B0：`8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`
- B1：`892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb`

当前机器搜索结果：

- 未找到这两个 SHA 对应的 `model.safetensors` 文件。
- Git worktree、`D:\` 本地搜索范围以及已有压缩包目录清单中，没有发现 B0/B1 VSPW 微调权重文件。
- `D:\segformer_b0_hf_cache.tar.gz` 只列出 `models--nvidia--segformer-b0-finetuned-ade-512-512/.../model.safetensors`，这是 ADE 预训练缓存，不能冒充 VSPW-124 checkpoint。
- 仓库 `results/R76R_E04_freeze/freeze_summary.json` 和 `source_provenance.json` 保存了 B0 的历史 model hash 及训练 checkpoint hash，但不包含可加载的训练权重本体；B1 只有对应实验记录/目标 hash。

结论：两个冻结 checkpoint 在当前本地环境中均未恢复。后续没有权重前，不执行新推理，不把 ADE checkpoint 当论文模型，不伪造 mVC 或新 Pareto 数字。

## 4. 环境与硬件

| 项目 | 历史实验 contract | 当前本机 |
|---|---|---|
| Python | 3.10.8（R11B preflight） | 3.14.4 系统 Python；仓库 `.venv` 为 3.12.10 |
| PyTorch | 2.1.2+cu118 | 系统 Python：2.11.0+cpu |
| Transformers | 4.44.2 | 未按历史 CUDA 环境复现 |
| CUDA/GPU | CUDA 11.8；R11A RTX 4090 D；R11B RTX 3080 Ti | `torch.cuda.is_available() = False`，GPU count 0 |
| 精度/输入 | BF16、512×512、batch 1 | 无 CUDA，无法执行正式 timing |

历史运行环境证据：`results/R11B_E00_preflight/preflight.json`、`results/R71_E00_environment/environment.json`、`reports/round11a_hcpr_state.json`。

## 5. 论文数据来源链审计

### Fixed policy

- Round 10 fixed quality：`results/R10_E03_quality/{b0,b1}/*.json`。
- Round 10 fixed latency：`results/R10_E04_latency/{b0,b1}.json`，Round 10 state 明确 GPU 为 RTX 4090 D。
- 论文 Table 3 的 `.825/2.057`（B0 S2-S4/FI-4）和 `1.280/1.952`（B1 S2-S4/FI-4）与 Round 10 artifact 一致。

### HCPR source device

- Round 11A HCPR 汇总：`reports/round11a_hcpr_quality_latency_table.csv`。
- `reports/round11a_hcpr_state.json` 指明 GPU RTX 4090 D、24,392 val frames、32 training videos、seed 42、age guard 7、完整 lambda grid。

### HCPR target device

- quality：`results/R11B_E03_transfer/{b0,b1}/summary.json`。
- latency：`results/R11B_E04_latency/{b0,b1}.json`。
- hardware profile：`results/R11B_E02_profile/{b0,b1}.json`，设备为 RTX 3080 Ti。

### 已确认的混用

1. 论文 Table 3 的 fixed values 是 Round 10 数字，表格没有 GPU 列。
2. 论文 source-device HCPR sweep 来自 Round 11A；因此 source fixed 与 HCPR 至少跨 Round/protocol。
3. `scripts/run_round11b_gate.py` 的 `fixed_frontier()` 从 Round 10 quality table 读 fixed damage，再把 Round 11B latency speedup 填进去；target fixed 点因此是“Round 10 quality + Round 11B latency”的混合来源。
4. 当前 artifact 可以重算 target 的 diagnostic count，但不能把它称为同一 quality/timing protocol 的严格 Pareto frontier。

详细逐点结果见 `reports/post_mmm/pareto_consistency_audit.md/json`。

## 6. 可重算与必须恢复的实验

### 无 GPU、无 checkpoint 也可完成

- Pareto/dominance 逻辑和来源链审计。
- Table 3、fixed/HCPR 数字的一致性检查。
- VSPW label 0/253/255 canonicalization contract 检查。
- 路由占比、stage-call ratio、已有 speedup CI 的汇总。
- Review evidence matrix、下一轮 protocol 和实验命令设计。

### 需要恢复冻结 checkpoint + 正确 CUDA 环境

- 重新推理 Full/fixed/HCPR 的 quality，尤其 target-device matched fixed quality。
- Two-action Full↔S2-S4 完整曲线。
- NewCost/OldCost/DriftOnly/ConstantCost 的独立 calibration 和 matched curves。
- age guard、motion subset、global/local drift 分析。
- per-frame masks、mVC8/mVC16、route/cache-age/per-frame IoU 日志和连续帧可视化。

### 需要重新训练的条件

只有当两个冻结 `model.safetensors` 无法从外部备份恢复，且下一稿必须复现原始 checkpoint 时，才进入重训。现有训练记录可帮助恢复配置，但不能证明重训会得到相同 bitwise model；重训前必须先保存新的 provenance、seed、数据 hash 和最终 SHA256。

## 7. 当前可复现性等级

**代码与历史汇总：中等；严格端到端复现：不可用。**

能做的事情是：运行 CPU 单元测试、读取已有 JSON/CSV、重算历史 Pareto 逻辑、审计 label contract。不能做的事情是：在当前机器上从冻结 checkpoint 重新得到论文 quality/timing，或计算 matched HCPR temporal consistency。

## 8. 阻塞项

1. B0/B1 VSPW-124 frozen `model.safetensors` 不在当前本地。
2. 当前机器无 CUDA GPU，无法执行 formal timing/inference。
3. R11B fixed quality 不是 target-device 同轮重新推理结果。
4. HCPR 历史实验没有保存 per-frame prediction masks。
5. two-action baseline 尚未实现。

## 9. 审计后的结论边界

- 可以保留：代码层面的 whole-stage cache/refresh 语义、forced-full equivalence、physical stage-call omission 的历史证据。
- 可以暂时保留但必须限 scope：两张已测 GPU 上的 profile-conditioned transfer 观察。
- 必须撤回或暂停：source-device “全部 HCPR 点 non-dominated”、严格 target fixed-vs-HCPR Pareto、progressive 独立增量已被证明、temporal consistency 已验证、hardware calibration 普遍有效。
