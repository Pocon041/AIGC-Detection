# AIGC Image Detection Experiments

这是一个暂未正式命名的 AIGC 图像检测 PyTorch 实验项目。当前主线是 **CC-FLED Phase 1**：先验证“局部可比真实先验”是否比全局真实先验更适合跨生成器检测，再决定是否进入完整 CC-FLED。

## 项目结构

```text
<project-root>/
  src/okk/          # 可复用模型、数据集、指标和 CC-FLED 工具
  scripts/          # 数据构建、预处理、训练、评估、分析脚本
  examples/         # AutoDL / 本地运行示例
  docs/             # 方案、实验记录、研究报告
  references/       # 论文 PDF
  manifests/        # 本地生成的数据清单，默认不进 git
```

生成物默认写入 `cache/`、`checkpoints/`、`outputs/`、`runs/`，这些目录已被 `.gitignore` 忽略。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

AutoDL 单张 RTX 5090 上建议先用 `batch_size=32` 缓存特征，显存确认充足后再调到 `64`。

## CC-FLED Phase 1

Phase 1 只回答一个核心问题：

> 在真实图像库中检索语义/结构代理变量可比的真实锚点后，局部 real prior 是否优于全局 real prior？

这里的 `z_c` 是 **audited proxy**，不是视觉复杂度真值。`edge density`、`spectral entropy`、`patch variance entropy` 等只作为操作性代理变量；分辨率、JPEG 量化、文件格式等 pipeline 变量会被缓存用于混杂审计，但默认不参与 local prior 条件化。

一键示例：

```bash
MANIFEST=manifests/dda_coco_paired.csv \
LOWLEVEL_CKPT=checkpoints/beyond_lowlevel_precomputed_best.pth \
BATCH_SIZE=32 \
bash examples/run_ccfled_phase1_autodl.sh
```

分步运行：

```bash
python scripts/preprocessing/cache_ccfled_features.py \
  --manifest manifests/dda_coco_paired.csv \
  --lowlevel-checkpoint checkpoints/beyond_lowlevel_precomputed_best.pth \
  --batch-size 32 \
  --out cache/ccfled_dda_coco.npz

python scripts/analysis/audit_complexity_proxies.py \
  --cache cache/ccfled_dda_coco.npz \
  --out outputs/ccfled_proxy_audit.json

python scripts/evaluation/evaluate_local_prior.py \
  --cache cache/ccfled_dda_coco.npz \
  --bank-split train \
  --eval-split val \
  --k-values 8,16,32,64 \
  --out outputs/ccfled_local_prior_val.json
```

## 主要脚本

```text
scripts/preprocessing/cache_ccfled_features.py   # 缓存 z_f / z_s / z_c_proxy
scripts/analysis/audit_complexity_proxies.py     # 审计 proxy 与 label / generator / pipeline 的混杂
scripts/evaluation/evaluate_local_prior.py       # 比较 global prior 与 local comparable prior
scripts/training/train_lowlevel_precomputed.py   # 训练 low-level pretext 主干
scripts/evaluation/fit_beyond_gmm.py             # 旧版 global GMM baseline
scripts/training/train_conditional.py            # 旧版 conditional detector baseline
```

## 评价重点

不要只看 AUROC。Phase 1 至少同时检查：

- `TPR@FPR=1%` 和 `TPR@FPR=5%`
- per-generator / per-operation 分组指标
- macro average 和 worst-group 表现
- KNN 邻域稳定性，默认 `K=8,16,32,64`
- proxy 是否强烈预测 label、generator、split 或后处理管线

如果 `semantic + proxy local prior` 只在单一数据来源上变好，不能声称复杂度建模有效，只能作为混杂线索继续审计。
