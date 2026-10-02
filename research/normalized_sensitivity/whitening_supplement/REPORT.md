# Whitening normalized representation sensitivity

完整复用第 12 项协议：2048 个固定 ImageNet 训练样本，每图 128 个固定 CUDA Gaussian 噪声，t 为 clean fraction，null class 1000，epoch200/step250200 raw model 权重，不用 EMA。24 个 norm1/norm2 输入，逐 patch 非仿射 RMS（eps=1e-6），无 GAP、无 head。

输入先从 uint8 转到 [-1,1]，用各 checkpoint 保存且完全相同的 basis/mean/scales 做 FP32 whitening，然后在 whiten 坐标加入 isotropic noise。未重新拟合 whitening。

B 为 N-1 的 between covariance trace 减去 W/128；W 使用 K-1 的 within covariance trace，不裁剪负修正值。下面 B/D 与 W/D 是 24 个观测点的均值；R 是 24 个点的 B/W 均值，并非表内均值 B 除以均值 W。

| Prediction | t clean | Mean B/D | Mean W/D | Mean event R | Last-event R |
|---|---:|---:|---:|---:|---:|
| whiten_clean | 0.25 | 0.09237689 | 0.27701141 | 0.833324 | 3.684248 |
| whiten_clean | 0.50 | 0.21981938 | 0.17325175 | 2.326533 | 8.115614 |
| whiten_clean | 0.75 | 0.34063385 | 0.07262050 | 6.516170 | 19.258453 |
| whiten_velocity | 0.25 | 0.08135280 | 0.28521690 | 0.836646 | 3.929386 |
| whiten_velocity | 0.50 | 0.17741572 | 0.14754370 | 2.448387 | 9.132240 |
| whiten_velocity | 0.75 | 0.25573598 | 0.04587019 | 8.641856 | 22.286566 |

## 同次运行的 clean/velocity 对比

| t clean | Clean R / velocity R | Clean W / velocity W | Clean B / velocity B |
|---|---:|---:|---:|
| 0.25 | 0.996029 | 0.971231 | 1.135510 |
| 0.50 | 0.950231 | 1.174240 | 1.239007 |
| 0.75 | 0.754024 | 1.583174 | 1.331975 |

## 历史未 whitening 参考

以下 plain 值来自原报告，未在本机重新运行。样本 IDs/labels 完全一致，但历史环境为 Torch2.13，本次为 Torch2.9.1，因此只作为历史参考，不声称跨环境严格复现实验或因果效应。

| Model | t clean | Whiten mean R | Historical plain mean R | Whiten / historical plain |
|---|---:|---:|---:|---:|
| whiten_clean | 0.25 | 0.833324 | 1.500811 | 0.555249 |
| whiten_clean | 0.50 | 2.326533 | 4.392907 | 0.529611 |
| whiten_clean | 0.75 | 6.516170 | 17.687284 | 0.368410 |
| whiten_velocity | 0.25 | 0.836646 | 1.042941 | 0.802199 |
| whiten_velocity | 0.50 | 2.448387 | 1.894492 | 1.292371 |
| whiten_velocity | 0.75 | 8.641856 | 3.903064 | 2.214121 |

## 结果解读

这里的 t 是 clean fraction，因此 t=0.75 比 t=0.25 的输入噪声更少。以下是同次运行的描述性比较，不代表统计显著性或生成质量。

- t=0.25：velocity 的逐点 R 均值相对 clean 高 0.40%；24 个位置中有 8 个 R 更高。
- t=0.50：velocity 的逐点 R 均值相对 clean 高 5.24%；24 个位置中有 12 个 R 更高。
- t=0.75：velocity 的逐点 R 均值相对 clean 高 32.62%；24 个位置中有 21 个 R 更高。

t=0.25 的深度平均 R 基本一致；t=0.50 velocity 略高；t=0.75 的差距最明显，且分布在大多数层。t=0.75 时 velocity 的平均 W/D 更低，但平均 B/D 也更低，因此不能把较高 R 解读为保留了更多绝对语义信息。

与原未 whitening 历史表对照时，clean 的 R 均值降低，而 velocity 在 t=0.50/0.75 更高；这只是跨历史实验的参考，不是严格控制环境和训练条件后的因果结论。

最后一个观测点是第 12 层 MLP 的输入，不是模型输出。

## 边界与运行记录

B 表示跨图变化，不等于纯语义；W 是 patch RMS 后的噪声敏感度，不能单独当作绝对幅值敏感度。较低 W 可能伴随表征塌缩，应同时检查 B 和 R。结果没有置信区间，不能从这些汇总量直接声称统计显著性。

纯 inference_mode、无优化器、无 W&B、未写原 checkpoint。clean/velocity 分别共用 GPU0/GPU4，PyTorch allocator cap 为每进程 2 GiB；CUDA runtime 额外占用不包含在 cap 内。并行的既有训练未被停止、重启或改配置。

统计直接复用原 streaming_stats.py；只编译 moment reduction，并在每项任务首个观测点与 eager moments 校验。全部 6 项 complete 和 144 条完整观测才纳入汇总。

- `summary.csv/json`：深度均值。
- `per_event.csv`：保留原始 B_raw、B_corrected、W、两个比值及每维结果。
- `data/*/`：每项 manifest、metrics、complete、reduction validation。
- `audit.json`：完整性、有限值与有限 K 修正检查。
- `figures/ratio_by_depth.png/pdf`、`figures/variance_by_depth.png/pdf`：独立图件。

![逐层 B/W](figures/ratio_by_depth.png)

![逐层 B 和 W](figures/variance_by_depth.png)
