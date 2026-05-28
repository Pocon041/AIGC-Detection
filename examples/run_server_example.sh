set -e

python scripts/data/make_manifest.py \
  --real-train /root/autodl-tmp/data/real_train \
  --fake-train /root/autodl-tmp/data/fake_train \
  --real-val /root/autodl-tmp/data/real_val \
  --fake-val /root/autodl-tmp/data/fake_val \
  --real-test /root/autodl-tmp/data/real_test \
  --fake-test /root/autodl-tmp/data/fake_test \
  --group custom \
  --fake-generator unknown_fake \
  --out manifests/custom.csv

python scripts/training/train_probe.py \
  --manifest manifests/custom.csv \
  --epochs 20 \
  --batch-size 64 \
  --out checkpoints/probe_best.pth

python scripts/evaluation/evaluate_probe.py \
  --manifest manifests/custom.csv \
  --split test \
  --checkpoint checkpoints/probe_best.pth \
  --out outputs/probe_eval.json

python scripts/data/make_pair_manifest.py \
  --real-dir /root/autodl-tmp/data/aligned_real_train \
  --fake-dir /root/autodl-tmp/data/aligned_fake_train \
  --split train \
  --group aligned \
  --operation aligned_fake \
  --generator unknown_fake \
  --out manifests/aligned_train.csv

python scripts/data/make_pair_manifest.py \
  --real-dir /root/autodl-tmp/data/aligned_real_val \
  --fake-dir /root/autodl-tmp/data/aligned_fake_val \
  --split val \
  --group aligned \
  --operation aligned_fake \
  --generator unknown_fake \
  --out manifests/aligned_val.csv

python - <<'PY'
import csv
from pathlib import Path
rows = []
for p in [Path('manifests/aligned_train.csv'), Path('manifests/aligned_val.csv')]:
    with p.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows.extend(list(reader))
with Path('manifests/aligned.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path','label','group','pair_id','mask_path','operation','generator','split'])
    writer.writeheader()
    writer.writerows(rows)
PY

python scripts/training/train_conditional.py \
  --manifest manifests/aligned.csv \
  --epochs 20 \
  --batch-size 32 \
  --condition hybrid \
  --residual beyond \
  --beyond-checkpoint checkpoints/beyond_lowlevel_best.pth \
  --out checkpoints/conditional_best.pth

python scripts/evaluation/evaluate_conditional.py \
  --manifest manifests/custom.csv \
  --split test \
  --checkpoint checkpoints/conditional_best.pth \
  --out outputs/conditional_eval.json
