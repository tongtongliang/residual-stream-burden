# Hidden Representation Noise Linear Probes

新增：[Group1 semantic-flow 补测](group1_semantic/REPORT.md)，DiP / PixelDiT / HyperDiT / DeCo，12/12 个任务已完成。[结果分析与统计](group1_semantic/ANALYSIS.md)，附原始数据、配置与代码快照。

实验数据与分析的统一归档位置。这里只组织已有结果，不启动训练、推理或 W&B。

完整 Pixel / RAE 对比：[GAP 实验汇总报告](REPORT.md)，包含第11层 Top1/Top5、逐层峰值、完整数值及曲线。

| 组别 | 内容 | 状态 |
|---|---|---|
| [Group 1](group1/REPORT.md) | Pixel JiT-clean / JiT-velocity / SiHC P4 sublayer velocity | 9 个任务完成 |
| [Group 2](group2/REPORT.md) | RAE DINOv2-B latent 上的 JiT / mHC，clean / velocity | 12 个任务完成，原始数据已校验归档 |
| [DINOv2-B baseline](dinov2_baseline/REPORT.md) | RAE 输入端 normalized DINOv2-B latent 的 noisy GAP probe | 4 个 probe 完成，epoch 40 全量验证 |

## 目录约定

- `groupN/REPORT.md`：协议、结果、分析与限制。
- `groupN/data/raw/<model>/noiseXX/`：原始数值及运行协议，保持原值。
- `groupN/data/accuracy.csv`：全部层、全部测试模式的汇总数值。
- `groupN/data/accuracy_by_seed.csv`：每个测试噪音实例的原始准确率。
- `groupN/data/training.csv`：从运行日志提取的逐 epoch loss 与耗时。
- `groupN/figures/`：可以从仓库内数据重绘的 PNG/PDF。
- `groupN/provenance/`：代码快照与来源清单，不包含模型权重或环境。
- `dinov2_baseline/`：不经过生成 backbone 的输入表征对照，不能当作某个 hidden layer。

后续 Group 沿用此结构。不得把 smoke 或未完成任务标成正式结果；输入噪音
系数使用 alpha，Group 1 的模型时间为 t=1-alpha，避免与模型训练目标 clean 混淆。

## 复现分析

在仓库根目录运行（仅依赖已有数据及 numpy/matplotlib）：

```bash
python analysis_reports/04_hidden_noise_linear_probe/scripts/render_group1.py
python analysis_reports/04_hidden_noise_linear_probe/scripts/render_comparison.py
```

`scripts/archive_group.py` 用于从外部运行目录导入已完成的 Group，不读取 checkpoint、
不计算哈希、不访问网络。代码快照只用于追溯，不是独立可运行的训练包。
