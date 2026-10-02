"""Regenerate the completed GAP comparison from archived CSV files."""
from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
rows = []
for group in (1, 2):
    with (ROOT / f"group{group}/data/accuracy.csv").open() as handle:
        rows.extend(
            dict(row, group=group)
            for row in csv.DictReader(handle)
            if row["mode"] == "matched_noise"
        )

models = list(dict.fromkeys(row["model"] for row in rows))
names = dict(
    zip(
        models,
        [
            "Pixel JiT-clean",
            "Pixel JiT-velocity",
            "Pixel SiHC-P4 velocity",
            "RAE JiT-clean",
            "RAE JiT-velocity",
            "RAE mHC-clean",
            "RAE mHC-velocity",
        ],
    )
)
assert len(models) == 7
alphas = [0.25, 0.5, 0.75]


def select(model, alpha):
    return sorted(
        [
            row
            for row in rows
            if row["model"] == model and float(row["alpha"]) == alpha
        ],
        key=lambda row: int(row["after_block"]),
    )


assert all(len(select(model, alpha)) == 11 for model in models for alpha in alphas)

with (ROOT / "dinov2_baseline/data/accuracy.csv").open() as handle:
    baseline_rows = list(csv.DictReader(handle))


def baseline_select(alpha, mode="matched_noise"):
    return next(
        row
        for row in baseline_rows
        if float(row["train_noise"]) == alpha and row["mode"] == mode
    )


def summary_table(selected, peak=False):
    output = [
        "| 模型 | 25%噪声 | 50%噪声 | 75%噪声 |",
        "|---|---:|---:|---:|",
    ]
    for model in selected:
        cells = []
        for alpha in alphas:
            data = select(model, alpha)
            row = max(data, key=lambda item: float(item["top1"])) if peak else data[-1]
            if peak:
                cells.append(f"{float(row['top1']):.2f}（层{row['after_block']}）")
            else:
                cells.append(f"{float(row['top1']):.2f} / {float(row['top5']):.2f}")
        output.append("| " + names[model] + " | " + " | ".join(cells) + " |")
    return "\n".join(output)


def depth_table(selected, alpha, metric):
    title = "Top-1" if metric == "top1" else "Top-5"
    output = [
        "| Block 后 | "
        + " | ".join(f"{names[model]} {title}" for model in selected)
        + " |",
        "|---:|" + "|".join("---:" for _ in selected) + "|",
    ]
    by_model = {model: select(model, alpha) for model in selected}
    for layer in range(1, 12):
        values = [
            f"{float(by_model[model][layer - 1][metric]):.2f}" for model in selected
        ]
        output.append(f"| {layer} | " + " | ".join(values) + " |")
    return "\n".join(output)


def baseline_table():
    output = [
        "| Probe 训练噪声 | 验证输入 | Top-1 (%) | Top-5 (%) |",
        "|---:|---|---:|---:|",
    ]
    for alpha in (0.0, 0.25, 0.5, 0.75):
        clean = baseline_select(alpha, "clean")
        output.append(
            f"| {int(alpha * 100)}% | clean | {float(clean['top1']):.3f} | "
            f"{float(clean['top5']):.3f} |"
        )
        if alpha:
            noisy = baseline_select(alpha)
            output.append(
                f"| {int(alpha * 100)}% | matched {int(alpha * 100)}% noise | "
                f"{float(noisy['top1']):.3f} +/- {float(noisy['top1_seed_std']):.3f} | "
                f"{float(noisy['top5']):.3f} +/- {float(noisy['top5_seed_std']):.3f} |"
            )
    return "\n".join(output)


for group in (1, 2):
    figure_dir = ROOT / f"group{group}/figures"
    figure_dir.mkdir(exist_ok=True)
    for metric in ("top1", "top5"):
        figure, axes = plt.subplots(
            1, 3, figsize=(15, 4), sharey=True, layout="constrained"
        )
        for axis, alpha in zip(axes, alphas):
            for model in models:
                data = select(model, alpha)
                if data[0]["group"] != group:
                    continue
                axis.plot(
                    range(1, 12),
                    [float(row[metric]) for row in data],
                    marker=".",
                    label=names[model],
                )
            axis.set(
                title=f"Input noise {int(alpha * 100)}%",
                xlabel="After block / next fan-in",
                xticks=range(1, 12),
            )
            axis.grid(alpha=0.2)
        axes[0].set_ylabel(metric + " accuracy (%)")
        axes[-1].legend(fontsize=8)
        figure.savefig(figure_dir / f"completed_gap_{metric}.png", dpi=160)
        plt.close(figure)


protocol = """## 统一读数协议

- 所有结果均为 matched_noise：噪声在冻结模型的输入端加入，然后 forward；不是在 hidden 或 GAP 后加噪。
- ImageNet validation 全部50,000张，固定中心裁剪、不 flip；三个验证噪声种子的准确率取平均。每个训练噪声水平有自己的 probe，不跨噪声水平共用。
- 输入 z=(1-alpha)x+alpha*epsilon，alpha=25/50/75%。Pixel x 为[-1,1]图片；RAE x 为 RAE.encode 归一化后的 DINOv2-B latent。Pixel 时间 t=1-alpha；RAE 时间 t=alpha。
- 冻结 epoch200 raw backbone；输入 null class1000，类别标签只用于监督/核对 probe。Probe 训练40个epoch，验证不更新模型或 probe。
- 普通 JiT 在 block1..11 更新后取下一层 fan-in。SiHC 是下一层 read 后、AdaLN 前的 workspace；mHC 是下一层聚合后的 fan-in。不能把后两者解释为高分辨率/完整多流残差状态。
- 每个 patch 做 RMS，再 GAP，每个边界有独立线性分类器。第11层不是第12层 backbone endpoint。
- 表中不混入辅助 clean-input 测试。原始 CSV 保留该模式并单独标记。验证种子 SD 不是独立训练重复的不确定性。
"""

findings = """## 汇总分析

1. Pixel 普通 velocity 的 GAP 类别可读性明显较弱；SiHC velocity workspace 的峰值与 clean 相当并略高。不过该结果比较的是不同结构/读取位置，不单独证明某一结构因素的因果作用。
2. RAE 普通 velocity 的末段 Top-1 比 clean 高约0.9--1.0个百分点，但 clean 的早期峰值并不低。不能将 pixel 域的 velocity 结论推广到 RAE。
3. mHC 不是普遍改善。相对普通 clean，mHC-clean 第11层提高约1.27/0.95/0.44个百分点；mHC-velocity 则比普通 velocity 低约0.82/1.10/2.30个百分点，75%噪声的末段下降最明显。
4. normalized DINOv2-B输入本身在matched noise下已有约80% Top-1；RAE hidden的绝对准确率主要继承这一强语义起点，重点应读深度变化和相对输入基线的增减。
5. 峰值表是同一验证曲线上的描述性最佳层，不是独立选层测试。未做独立训练重复，不声称统计显著。

## 与 patch Gram 分析的区别

这里测跨图片的类别线性可读性；[残差 patch geometry](../05_jit_patch_geometry/REPORT.md) 测单张图片内部空间关系。Velocity 有一定 patch 对齐、却有较低 pixel GAP 分类准确率，并不矛盾。两项结果保留独立解释，不把低 GAP 分数称为“没有语义”。
"""

comparison = (
    """# Pixel 与 RAE：GAP linear probe 完整汇总

更新：2026-09-14。Group1的9个任务、Group2的12个任务和DINOv2-B输入基线均已完成。以下为归档原始数值自动生成，不是手工录入。

## 实验问题与测量对象

本实验测量：在固定噪声输入下，冻结生成backbone不同深度的patch表征经过逐patch RMS和空间GAP后，ImageNet类别能被线性读出的程度。它不是patch Gram/CKA分析，也不是用生成质量间接衡量语义。

- 普通JiT：第1--11个block更新后的residual stream，即下一层fan-in。
- SiHC：下一层channel-wise read得到的workspace，位置在AdaLN之前；不是完整高分辨率carrier。
- mHC：下一层mHC聚合后的workspace/fan-in；不是拼接的四流carrier。
- DINOv2-B baseline：RAE输入端normalized latent直接GAP，不经过JiT/mHC backbone。

## 固定第11层：Top-1 / Top-5（%）

"""
    + summary_table(models)
    + "\n\n## 第1--11层最高 Top-1（括号为层位置）\n\n"
    + summary_table(models, True)
    + "\n\n"
    + protocol
    + "\n"
    + findings
    + "\n## DINOv2-B输入表征基线\n\n"
    + baseline_table()
    + """

这是每个噪声水平独立训练的probe。matched-noise行才与RAE hidden probe主结果对应；clean行只是噪声训练probe到干净输入的辅助迁移结果。完整设置和原始数值见[DINOv2-B baseline](dinov2_baseline/REPORT.md)。

## 逐层曲线

![Pixel Top1](group1/figures/completed_gap_top1.png)

![RAE Top1](group2/figures/completed_gap_top1.png)

![Pixel Top5](group1/figures/completed_gap_top5.png)

![RAE Top5](group2/figures/completed_gap_top5.png)

## 数值与复现

- `group1/data/`、`group2/data/`包含全部层的汇总、逐验证噪声实例、训练记录和原始JSON；`dinov2_baseline/data/`保存输入表征基线。
- `group2/provenance/`保留代码及来源记录；四个任务的双卡续跑保持全局microbatch验证噪声序列，详情见各任务`paired_resume.json`与`PAIRED_GROUP2.md`。数值并非承诺与单卡训练逐位相同。
- 仅从仓库内数据重绘：`python research/linear_probes/scripts/render_comparison.py`。
- 本次整理未训练、推理、访问W&B或复制checkpoint。
"""
)
(ROOT / "REPORT.md").write_text(comparison)

group2 = (
    """# Group2：RAE DINOv2-B GAP linear probe

12/12个任务完成，probe均训练40个epoch并完成50k验证。完整跨域对比见[汇总报告](../REPORT.md)。

## 第11层 Top-1 / Top-5（%）

"""
    + summary_table(models[3:])
    + "\n\n## 各层最高 Top-1\n\n"
    + summary_table(models[3:], True)
    + "\n\n"
    + protocol
    + "\n## 完整逐层 Top-1\n"
    + "\n".join(
        f"\n### 输入噪音 {int(alpha * 100)}%\n\n"
        + depth_table(models[3:], alpha, "top1")
        for alpha in alphas
    )
    + """

## 完整逐层 Top-5
"""
    + "\n".join(
        f"\n### 输入噪音 {int(alpha * 100)}%\n\n"
        + depth_table(models[3:], alpha, "top5")
        for alpha in alphas
    )
    + """

## 结果解读

普通velocity的末段优于clean；mHC-clean改善末段可读性，但mHC-velocity的末段退化，高噪声最明显。mHC-velocity三个噪声水平的峰值都在第一处测量边界，不能以其早期分数代替末段表现。完整解释和限制见上级汇总报告。

![Top1](figures/completed_gap_top1.png)

![Top5](figures/completed_gap_top5.png)

归档校验：264行汇总、528行逐次验证、480条完整训练epoch；Top1/Top5均从原始四次验证记录重新计算核对。主结果只平均三个matched-noise视图，clean视图单列。双卡续跑记录与代码见`data/raw/`和`provenance/`。RAE输入端的直接probe对照见[DINOv2-B baseline](../dinov2_baseline/REPORT.md)。
"""
)
(ROOT / "group2/REPORT.md").write_text(group2)

print("Wrote combined report, Group2 full layer tables, and four depth curves.")
