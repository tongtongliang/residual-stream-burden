# Group 6: Whitening 后的类别线性可读性

## 已完成结果

两模型 × 三档噪声全部完成。冻结 epoch200/step250200 的 JiT-B/16 raw `model`
权重，独立线性分类头训练40轮，验证集50,000张，三次匹配噪声验证。
下表为已测11个位置中的描述性 Top-1 峰值，不是独立测试集选层后的无偏成绩。

| 噪声 q | Whitening clean | Whitening velocity | 原始 RGB velocity | 原始 RGB clean |
|---|---:|---:|---:|---:|
| 0.25 | 7.826% (block7) | 8.433% (block9) | 7.547% | 32.132% |
| 0.50 | 10.875% (block7) | 8.755% (block9) | 6.804% | 35.053% |
| 0.75 | 10.633% (block7) | 7.413% (block9) | 5.145% | 27.351% |

RGB对照来自 [Group1](../group1/REPORT.md)，不是本次重跑。
完整原值见 data/accuracy.csv；逐测试种子见 data/accuracy_by_seed.csv。

![matched-noise accuracy](figures/matched_noise_depth.png)

## 协议

- 不是 PCA128/64 截断：使用 checkpoint 自带的完整768维可逆 patch whitening。
  RGB uint8 先归一化到 [-1,1]，再按 P16 的 C,H,W 展平次序白化。
- `z=(1-q)*whiten(x)+q*epsilon`，epsilon 是白化坐标中的独立标准高斯，
  q=0.25/0.50/0.75；模型输入时间 `t_clean=1-q`。先白化，再加噪。
- 全量 ImageNet train 1,281,167张，ADM256 center crop，无随机翻转。
  null class；类别标签只给分类头，不给冻结骨干。
- 取完整 block1–11 更新后的下一 block fan-in，在 AdaLN 前。
  每个空间 patch 非仿射 RMS(eps=1e-6)，随后空间 GAP。
  每层独立768→1000带bias线性头，初始权重和bias均为零。
- microbatch128，全局有效 batch4096；AdamW 默认betas/eps，wd0，
  40轮 cosine LR 0.01→0.0001。骨干BF16 autocast，白化/RMS/分类头FP32，TF32关闭。
- 每轮随机排列 seed42+epoch_index；高斯噪声 seed700001+epoch_index。
  骨干在线前向，噪声不缓存。
- 主验证用与训练相同q，种子800001/800002/800003，报告均值及总体SD。
  `mode=clean` 是噪声训练头对 q=0、t_clean=1 的辅助迁移测试，不是clean-trained probe。
  SD仅衡量测试噪声变化，不代表独立训练seed不确定性。

## 实际执行与数值差异

- Clean：q=.25 前4轮为单卡eager，随后4卡compile；q=.50/.75 前14轮单卡compile，
  随后各2卡compile。均从最近完整epoch恢复头权重和AdamW，未保存轮重跑。
- Velocity：三个q从零开始，各2卡，全部compile default；GPU0–5，未使用6/7。
- 经研究者明确授权，取消compile相对FP32误差阈值导致的eager回退；保留有限值检查。
  Clean q=.25 是混合eager/compiled训练轨迹，不能称其全程compiled或数值完全等价。
- 分布式按完整128样本microbatch轮流分配，各rank按相同次序生成并跳过非本rank噪声，
  保持原始全局样本/噪声配对。梯度按实际全局样本数归一化后SUM归约，无样本补齐或丢弃。
  浮点归约顺序变化，结果不承诺逐位等价。验证整数命中数跨rank求和。
- 原始 protocol.json 的 loader_policy 是旧模板文字；本实验实际用像素DataLoader，
  并未用latent固定ring。保留原文件以追溯，以代码快照及本说明为准。
- provenance/code 是归档时源码快照；distributed_probe.py 此时已增加可选resume、
  从零初始化及元数据字段。Clean完成时使用其此前仅resume版本，见原始运行元数据。

## 解读与边界

1. Whitening velocity 相比原始RGB velocity，三档的峰值及11层平均Top-1都更高。
   但不意味着whitening总体有益：whitening clean相对原始clean大幅下降。
2. Whitening clean在中高噪声下峰值更高，velocity在低噪声下峰值略高。
   Clean峰值在block7，velocity在block9；block11则velocity三档均领先。
   只用最后一层不能代表完整深度曲线。
3. 这些结果说明差的生成表现不能简单等同于完全没有类别表征；但只测了
   RMS→GAP线性可读性，不是总语义信息，也不是生成质量的充分解释。
4. 同一q在RGB和白化坐标对应不同物理噪声。跨空间比较不是同样受损图片的严格因果对照。
5. [FID及embedding矩阵分析](../../14_whitening_patch_embedding/README.md) 使用生成评估EMA，
   本probe使用raw骨干，不能将两者当作同一权重状态的直接因果关联。

## 数据与重绘

- data/accuracy.csv：132行，六任务×11层×两种验证模式。
- data/accuracy_by_seed.csv：264行，六任务×11层×四个验证实例。
- data/training.csv：2640行，六任务×40轮×11层；从完成checkpoint的history提取。
- data/raw：原始JSON及执行协议；provenance/sources.json：骨干/分类头来源。
- 不包含checkpoint、数据集、环境或凭据。本次归档没有运行训练/推理，也未联系W&B。

从任意工作目录执行 `python /path/to/repo/analysis_reports/04_hidden_noise_linear_probe/group6/render.py`。
仅依赖numpy/matplotlib及仓库内CSV，不需要GPU或服务器绝对路径。
