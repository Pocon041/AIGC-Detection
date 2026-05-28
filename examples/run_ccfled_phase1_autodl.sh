set -e

MANIFEST=${MANIFEST:-manifests/dda_coco_paired.csv}
LOWLEVEL_CKPT=${LOWLEVEL_CKPT:-checkpoints/beyond_lowlevel_precomputed_best.pth}
CACHE=${CACHE:-cache/ccfled_phase1_features.npz}
BATCH_SIZE=${BATCH_SIZE:-32}

python scripts/preprocessing/cache_ccfled_features.py \
  --manifest "$MANIFEST" \
  --lowlevel-checkpoint "$LOWLEVEL_CKPT" \
  --batch-size "$BATCH_SIZE" \
  --out "$CACHE"

python scripts/analysis/audit_complexity_proxies.py \
  --cache "$CACHE" \
  --out outputs/ccfled_proxy_audit.json

python scripts/evaluation/evaluate_local_prior.py \
  --cache "$CACHE" \
  --bank-split train \
  --eval-split val \
  --k-values 8,16,32,64 \
  --out outputs/ccfled_local_prior_val.json
