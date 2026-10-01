# Pixel 与 RAE：GAP linear probe 完整汇总

更新：2026-09-14。Group1的9个任务、Group2的12个任务和DINOv2-B输入基线均已完成。以下为归档原始数值自动生成，不是手工录入。

## 实验问题与测量对象

本实验测量：在固定噪声输入下，冻结生成backbone不同深度的patch表征经过逐patch RMS和空间GAP后，ImageNet类别能被线性读出的程度。它不是patch Gram/CKA分析，也不是用生成质量间接衡量语义。

- 普通JiT：第1--11个block更新后的residual stream，即下一层fan-in。
- SiHC：下一层channel-wise read得到的workspace，位置在AdaLN之前；不是完整高分辨率carrier。
- mHC：下一层mHC聚合后的workspace/fan-in；不是拼接的四流carrier。
- DINOv2-B baseline：RAE输入端normalized latent直接GAP，不经过JiT/mHC backbone。

## 固定第11层：Top-1 / Top-5（%）

| 模型 | 25%噪声 | 50%噪声 | 75%噪声 |
|---|---:|---:|---:|
| Pixel JiT-clean | 25.32 / 46.45 | 27.54 / 49.64 | 22.82 / 43.91 |
| Pixel JiT-velocity | 5.38 / 14.51 | 4.56 / 12.68 | 3.47 / 10.17 |
| Pixel SiHC-P4 velocity | 25.51 / 46.68 | 27.81 / 50.32 | 22.88 / 43.98 |
| RAE JiT-clean | 80.15 / 95.47 | 80.09 / 95.41 | 80.06 / 95.36 |
| RAE JiT-velocity | 81.09 / 95.74 | 81.07 / 95.81 | 80.93 / 95.77 |
| RAE mHC-clean | 81.42 / 95.98 | 81.04 / 95.78 | 80.50 / 95.56 |
| RAE mHC-velocity | 80.27 / 95.38 | 79.98 / 95.26 | 78.63 / 94.76 |

## 第1--11层最高 Top-1（括号为层位置）

| 模型 | 25%噪声 | 50%噪声 | 75%噪声 |
|---|---:|---:|---:|
| Pixel JiT-clean | 32.13（层8） | 35.05（层8） | 27.35（层9） |
| Pixel JiT-velocity | 7.55（层9） | 6.80（层9） | 5.14（层9） |
| Pixel SiHC-P4 velocity | 33.75（层7） | 36.18（层7） | 27.80（层8） |
| RAE JiT-clean | 81.62（层3） | 81.29（层1） | 81.06（层1） |
| RAE JiT-velocity | 81.09（层11） | 81.07（层11） | 80.93（层11） |
| RAE mHC-clean | 81.58（层5） | 81.47（层5） | 80.93（层10） |
| RAE mHC-velocity | 81.27（层1） | 81.18（层1） | 80.88（层1） |

## 统一读数协议

- 所有结果均为 matched_noise：噪声在冻结模型的输入端加入，然后 forward；不是在 hidden 或 GAP 后加噪。
- ImageNet validation 全部50,000张，固定中心裁剪、不 flip；三个验证噪声种子的准确率取平均。每个训练噪声水平有自己的 probe，不跨噪声水平共用。
- 输入 z=(1-alpha)x+alpha*epsilon，alpha=25/50/75%。Pixel x 为[-1,1]图片；RAE x 为 RAE.encode 归一化后的 DINOv2-B latent。Pixel 时间 t=1-alpha；RAE 时间 t=alpha。
- 冻结 epoch200 raw backbone；输入 null class1000，类别标签只用于监督/核对 probe。Probe 训练40个epoch，验证不更新模型或 probe。
- 普通 JiT 在 block1..11 更新后取下一层 fan-in。SiHC 是下一层 read 后、AdaLN 前的 workspace；mHC 是下一层聚合后的 fan-in。不能把后两者解释为高分辨率/完整多流残差状态。
- 每个 patch 做 RMS，再 GAP，每个边界有独立线性分类器。第11层不是第12层 backbone endpoint。
- 表中不混入辅助 clean-input 测试。原始 CSV 保留该模式并单独标记。验证种子 SD 不是独立训练重复的不确定性。

## 汇总分析

1. Pixel 普通 velocity 的 GAP 类别可读性明显较弱；SiHC velocity workspace 的峰值与 clean 相当并略高。不过该结果比较的是不同结构/读取位置，不单独证明某一结构因素的因果作用。
2. RAE 普通 velocity 的末段 Top-1 比 clean 高约0.9--1.0个百分点，但 clean 的早期峰值并不低。不能将 pixel 域的 velocity 结论推广到 RAE。
3. mHC 不是普遍改善。相对普通 clean，mHC-clean 第11层提高约1.27/0.95/0.44个百分点；mHC-velocity 则比普通 velocity 低约0.82/1.10/2.30个百分点，75%噪声的末段下降最明显。
4. normalized DINOv2-B输入本身在matched noise下已有约80% Top-1；RAE hidden的绝对准确率主要继承这一强语义起点，重点应读深度变化和相对输入基线的增减。
5. 峰值表是同一验证曲线上的描述性最佳层，不是独立选层测试。未做独立训练重复，不声称统计显著。

## 与 patch Gram 分析的区别

这里测跨图片的类别线性可读性；[残差 patch geometry](../05_jit_patch_geometry/REPORT.md) 测单张图片内部空间关系。Velocity 有一定 patch 对齐、却有较低 pixel GAP 分类准确率，并不矛盾。两项结果保留独立解释，不把低 GAP 分数称为“没有语义”。

## DINOv2-B输入表征基线

| Probe 训练噪声 | 验证输入 | Top-1 (%) | Top-5 (%) |
|---:|---|---:|---:|
| 0% | clean | 79.372 | 95.036 |
| 25% | clean | 79.972 | 95.288 |
| 25% | matched 25% noise | 79.872 +/- 0.037 | 95.267 +/- 0.011 |
| 50% | clean | 80.850 | 95.710 |
| 50% | matched 50% noise | 80.485 +/- 0.072 | 95.545 +/- 0.011 |
| 75% | clean | 82.274 | 96.392 |
| 75% | matched 75% noise | 80.820 +/- 0.088 | 95.847 +/- 0.042 |

这是每个噪声水平独立训练的probe。matched-noise行才与RAE hidden probe主结果对应；clean行只是噪声训练probe到干净输入的辅助迁移结果。完整设置和原始数值见[DINOv2-B baseline](dinov2_baseline/REPORT.md)。

## 逐层曲线

![Pixel Top1](group1/figures/completed_gap_top1.png)

![RAE Top1](group2/figures/completed_gap_top1.png)

![Pixel Top5](group1/figures/completed_gap_top5.png)

![RAE Top5](group2/figures/completed_gap_top5.png)

## 数值与复现

- `group1/data/`、`group2/data/`包含全部层的汇总、逐验证噪声实例、训练记录和原始JSON；`dinov2_baseline/data/`保存输入表征基线。
- `group2/provenance/`保留代码及来源记录；四个任务的双卡续跑保持全局microbatch验证噪声序列，详情见各任务`paired_resume.json`与`PAIRED_GROUP2.md`。数值并非承诺与单卡训练逐位相同。
- 仅从仓库内数据重绘：`python analysis_reports/04_hidden_noise_linear_probe/scripts/render_comparison.py`。
- 本次整理未训练、推理、访问W&B或复制checkpoint。
