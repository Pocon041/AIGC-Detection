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
  --audit-split train \
  --out outputs/ccfled_proxy_audit.json

python scripts/evaluation/evaluate_local_prior.py \
  --cache "$CACHE" \
  --bank-split train \
  --eval-split val \
  --k-values 8,16,32,64 \
  --semantic-weight 1.0 \
  --proxy-weight 1.0 \
  --out outputs/ccfled_local_prior_val.json

python scripts/evaluation/fit_feature_scorer.py \
  --cache "$CACHE" \
  --scorer local_semantic_proxy \
  --bank-split train \
  --eval-split val \
  --k 32 \
  --semantic-weight 1.0 \
  --proxy-weight 1.0 \
  --model-out checkpoints/feature_scorer_local_semantic_proxy.joblib \
  --out outputs/feature_scorer_local_semantic_proxy_val.json
