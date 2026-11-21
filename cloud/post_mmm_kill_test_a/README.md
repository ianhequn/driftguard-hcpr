# POSTMMM-R1 / Kill Test A

这是一个已准备、未执行的云端控制包。目标是同一 GPU、同一 B0/B1 checkpoint、同一 VSPW manifest、同一 git commit 和同一 `POSTMMM-R1` protocol 下，重跑 fixed full grid 与 HCPR grid，判断 HCPR 是否真的提供 fixed 不具备的 quality-latency operating set。

## 先决条件

1. 云端 Python/CUDA 环境按 `requirements.txt` 或 `environment.yml` 创建。
2. 上传当前仓库的 source commit；不要只上传本目录。
3. 提供真实 `model.safetensors` 和 VSPW root。每个模型先用 `preflight.py` 写入 checkpoint SHA256、git commit、CUDA 和环境信息。
4. B0/B1 必须分别通过 exact SHA256；ADE base checkpoint 不能作为替代。
5. 运行前把 dataset manifest SHA256、calibration lambda grid 和目标 GPU 写入结果元数据。

## 本地 dry-run

```bash
cd cloud/post_mmm_kill_test_a
bash commands.sh
```

dry-run 只校验参数和输出计划，不导入 GPU 推理、不下载、不训练。

## 正式运行边界

`--execute` 入口默认 fail-closed。fixed latency 不允许调用历史 Round 10/11 latency；HCPR 不允许复用旧设备 profile。必须先把 target-device runner 和 `POSTMMM-R1` calibration artifact 绑定到 source commit，再打开执行。这样不会把旧 latency 或混合 quality/latency 拼成新 Pareto。

固定网格为 `S4/S3-S4/S2-S4 × FI-1/2/3/4/6/8`；HCPR quantiles 为 `Q20/Q35/Q50/Q65/Q80/Q90/Q95`。每个点必须保存 absolute latency、speedup、video-level 95% CI、mIoU damage、stage calls/frame、route counts、device、checkpoint SHA、dataset manifest SHA、git commit 和 protocol ID。

当所有行满足同一 contract 后，用 `finalize_pareto.py`；它会拒绝 mixed device/protocol/checkpoint/dataset rows。输出应放在新的 `POSTMMM-R1` 结果目录，不覆盖任何 Round 10/11 artifact。

## 上传/取回示例

```bash
scp -r post_mmm_kill_test_a USER@HOST:/workspace/
scp -r USER@HOST:/workspace/postmmm-r1-results ./results/
```

示例路径是占位符；本地不执行上传，也不下载或覆盖历史文件。
