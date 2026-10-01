"""Render Group1 figures and report from bundled CSV, without model execution."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'group1'
MODELS = [('plain_jit_clean', 'JiT-clean', '#2878b5'),
          ('plain_jit_velocity', 'JiT-velocity', '#d55e00'),
          ('sihc_sublayer_p4', 'SiHC-velocity', '#009e73')]
ALPHAS = [.25, .5, .75]


def main():
    with (ROOT/'data/accuracy.csv').open() as f:
        rows = list(csv.DictReader(f))
    data = {}
    for model, _, _ in MODELS:
        for alpha in ALPHAS:
            selected = sorted([r for r in rows if r['model'] == model and float(r['alpha']) == alpha
                               and r['mode'] == 'matched_noise'], key=lambda r: int(r['after_block']))
            assert [int(r['after_block']) for r in selected] == list(range(1, 12))
            data[model, alpha] = {key: np.array([float(r[key]) for r in selected])
                for key in ['top1', 'top5', 'top1_seed_std', 'top5_seed_std']}
    folder = ROOT/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey='row')
    x = np.arange(1, 12)
    for row_index, metric in enumerate(['top1', 'top5']):
        for column, alpha in enumerate(ALPHAS):
            ax = axes[row_index, column]
            for model, label, color in MODELS:
                y = data[model, alpha][metric]
                sd = data[model, alpha][metric+'_seed_std']
                ax.plot(x, y, color=color, marker='o', markersize=3, label=label)
                ax.fill_between(x, y-sd, y+sd, color=color, alpha=.15)
            ax.set_ylim(0, 40 if metric == 'top1' else 65)
            ax.set_xticks(x)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if row_index == 0:
                ax.set_title(f'Input noise alpha = {alpha:.2f}')
            else:
                ax.set_xlabel('After block (next fan-in / workspace)')
            if column == 0:
                ax.set_ylabel('Top-1 (%)' if metric == 'top1' else 'Top-5 (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ['png', 'pdf']:
        fig.savefig(folder/f'matched_noise_depth.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    text = '''# Group 1：噪音输入下的 Hidden Representation Linear Probe

## 状态与核心结论

9 个正式任务全部完成：3 个模型 × 3 个输入噪音水平，每个 probe 训练 40 epoch，
在完整 ImageNet 50k 验证集上评估。Group 2 仍暂停，不包含在本报告中。

**当前协议下，SiHC-velocity 的 workspace 中可线性读出的类别信息远强于普通
JiT-velocity，峰值接近或略高于 JiT-clean。优势集中于前中层，后段并非始终领先。**

## 实验协议

- 模型：200 epoch 的 JiT-B clean、JiT-B velocity、SiHC P4 sublayer velocity；
  使用 raw `model` 权重而非 EMA，checkpoint 标识见 `provenance/sources.json`。
- 对输入像素加噪：`z=(1-alpha)*x+alpha*epsilon`，`x` 在 [-1,1]，
  `epsilon ~ N(0,I)`，alpha 为 0.25 / 0.50 / 0.75；Group 1 模型时间为 `t=1-alpha`。
- 训练与主验证均先加噪，再送入冻结模型。模型使用 null class，真实类别只用于 probe。
- 边界为第 1–11 个 block 完整更新后。普通 JiT 取下一 block 的 fan-in；
  SiHC 取下一 attention 的 read workspace，在 AdaLN 前，不是高分辨率 carrier。
- 各 patch 做非仿射 RMS（eps=1e-6），然后空间 GAP。每个模型/噪音/边界有独立的
  768→1000 带 bias 线性分类器；不对 hidden feature 再加噪。
- 全量 train 1,281,167 张，ADM center crop、无 random flip；全局 batch 4096，
  microbatch 128，40 epoch，AdamW 默认 betas/eps、wd=0，cosine LR 0.01→0.0001。
- backbone BF16 autocast，pooling 与 probe FP32；compile default；backbone 无梯度。
- 在线生成新高斯噪音，不读取此前废弃的固定噪音特征缓存。
- 主验证：50k 图像，3 个独立测试噪音实例，alpha 和训练一致。
  均值与 SD 仅描述测试噪音变动，不是重复训练的统计不确定性。

### 六卡续跑说明

SiHC 75% 从已保存的第 5 epoch 恢复 probe 和 AdamW 状态，使用 6 卡完成余下训练
与验证。全局 batch、全局 shuffle 和样本覆盖不变；局部 loss 按全局样本数归一化，
梯度 all-reduce SUM。新噪音采用 rank-specific CUDA RNG，分布不变，但与单卡不是
逐元素相同的噪音。该限制适用于训练及验证，见 `provenance/SIX_GPU_RESUME.md`。

## 主结果：匹配输入噪音

下表分别列出 Top-1 / Top-5 在已测层中的描述性峰值。括号是 block 后的边界编号，
不是在独立测试集上重新验证过的“最佳层选择”。

'''
    for metric, label in [('top1', 'Top-1'), ('top5', 'Top-5')]:
        text += f'### {label} 峰值（%）\n\n| 输入噪音 | JiT-clean | JiT-velocity | SiHC-velocity |\n|---|---:|---:|---:|\n'
        for alpha in ALPHAS:
            values = []
            for model, _, _ in MODELS:
                y = data[model, alpha][metric]
                i = int(y.argmax())
                values.append(f'{y[i]:.2f}（{i+1}）')
            text += f'| {alpha:.0%} | '+ ' | '.join(values)+' |\n'
        text += '\n'
    text += '''## 逐层曲线

![各噪音水平下的 Top-1 和 Top-5](figures/matched_noise_depth.png)

阴影为三个测试噪音实例的 ±1 SD，不是置信区间。可下载 [PDF](figures/matched_noise_depth.pdf)。

## 分析

1. **velocity 目标本身不足以解释弱线性可读性。** 普通 JiT-velocity 峰值 Top-1
   只有约 5–8%，而同为 velocity 目标的 SiHC 达到约 28–36%。这说明在当前训练结果
   和 probe 协议下，两者的类别信息可读性不同；不等于普通模型没有语义信息。
2. **相对 JiT-clean，SiHC 的主要变化是更早出现较强表征。** 在第 5 个 block 后，
   三个噪音水平的 Top-1 提升分别为 7.90、8.83、6.68 个百分点。峰值位于约第 7–8 层，
   JiT-clean 位于约第 8–9 层。相邻层有近似平台，不应把单层峰值过度解释为精确转折。
3. **后段仍然回落。** JiT-clean 在第 9–10 层优于 SiHC；第 11 层二者很接近。
   SiHC 并没有在所有深度都更好，也没有消除晚层类别线性可读性的下降。
4. **高噪音仍有损害。** SiHC 峰值 Top-1 从 50% 噪音的 36.18% 降至 75% 的 27.80%。
   相对 JiT-clean 的峰值优势为 1.62、1.13、0.45 个百分点。不能称为噪音不变表征。
5. **Top-5 支持相同趋势。** SiHC 峰值分别为 57.19%、60.50%、50.35%，相比 JiT-clean
   高 1.58、1.37、0.60 个百分点。25% 下 SiHC 的 Top-5 峰值在第 8 层，与 Top-1
   的第 7 层不完全相同，因此两项指标分别选取描述性峰值。

## 结论边界

- 测到的是 null conditioning 下 patch RMS→GAP 的线性可读性，不是语义信息总量，
  也不是任意非线性读出能达到的最佳准确率；没有通过超参数搜索证明 probe 已达最优。
- SiHC 与 plain 模型的 patch 结构和残差拓扑同时不同，且 SiHC 测的是 read 后 workspace。
  本实验不能单独识别拓扑的因果贡献，不能直接证明 read 降噪、transport burden 降低，
  或推出更好的 gFID。峰值的小幅优势也尚无独立训练 seed 的重复验证。
- 文件中的 `mode=clean` 是额外的“噪音训练 probe → 干净输入、干净时间”迁移测试。
  主图和主表没有混入它；它不是 clean-trained probe，不能代表干净表征自身的可分类性。
- JiT-clean 名称指原 diffusion 模型的预测目标，和 validation 是否加噪是两件事。

## 完整逐层数据

'''
    for alpha in ALPHAS:
        text += f'### 输入噪音 {alpha:.0%}\n\n| Block 后 | clean Top-1 | velocity Top-1 | SiHC Top-1 | clean Top-5 | velocity Top-5 | SiHC Top-5 |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for i in range(11):
            values = [data[model, alpha][metric][i] for metric in ['top1', 'top5'] for model, _, _ in MODELS]
            text += f'| {i+1} | '+' | '.join(f'{v:.2f}' for v in values)+' |\n'
        text += '\n'
    text += '''## 数据与可复现性

- [汇总 CSV](data/accuracy.csv)：198 行，包含主验证和辅助 clean 模式的原始精度及 SD。
- [逐测试实例 CSV](data/accuracy_by_seed.csv)：396 行，保留 Top-1 / Top-5 原始数值。
- [逐 epoch 训练 CSV](data/training.csv)：360 行，包含 loss、耗时及单卡/多卡标识。
- `data/raw/` 原样保存 9 个任务的数值、原协议和编译检查；六卡协议补充单独保留，
  不能只读最初单卡 protocol.json 而忽略 distributed_resume.json。
- `provenance/` 包含代码快照与外部路径清单，checkpoint 和完整日志不打包。
  代码快照依赖原服务器模型仓库，仅用于追溯；图表重绘不依赖这些外部路径。
- 导入检查：9 个完成标记、50k 测试规模、11 个边界、4 个测试实例、数值有限且
  在 [0,100]、Top-5≥Top-1，并由逐实例值重新计算均值/SD核对原汇总。

从仓库根目录运行 `python analysis_reports/04_hidden_noise_linear_probe/scripts/render_group1.py`
即可重绘本报告与图表，无需 GPU、checkpoint、网络或 W&B。
'''
    (ROOT/'REPORT.md').write_text(text)
    print(f'Rendered {ROOT / "REPORT.md"}')


if __name__ == '__main__':
    main()
