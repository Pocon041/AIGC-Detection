$ErrorActionPreference = "Stop"

python scripts\data\make_manifest.py `
  --real-train "D:\path\to\real_train" `
  --fake-train "D:\path\to\fake_train" `
  --real-val "D:\path\to\real_val" `
  --fake-val "D:\path\to\fake_val" `
  --real-test "D:\path\to\real_test" `
  --fake-test "D:\path\to\fake_test" `
  --group "custom" `
  --fake-generator "unknown_fake" `
  --out "manifests\custom.csv"

python scripts\training\train_probe.py `
  --manifest "manifests\custom.csv" `
  --epochs 20 `
  --batch-size 64 `
  --out "checkpoints\probe_best.pth"

python scripts\evaluation\evaluate_probe.py `
  --manifest "manifests\custom.csv" `
  --split test `
  --checkpoint "checkpoints\probe_best.pth" `
  --out "outputs\probe_eval.json"
