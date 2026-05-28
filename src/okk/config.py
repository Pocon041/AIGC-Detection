from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentConfig:
    project_root: Path = PROJECT_ROOT
    manifest_dir: Path = PROJECT_ROOT / "manifests"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"
    cache_dir: Path = PROJECT_ROOT / "cache"
    image_size: int = 224
    batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 8
    seed: int = 2026
    device: str = "cuda"
    backbone_name: str = "timm/vit_base_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    feature_dim: int = 768
    probe_hidden_dim: int = 256
    detector_hidden_dim: int = 256
    detector_depth: int = 2
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    margin: float = 0.2
    lambda_nll: float = 1.0
    lambda_rank: float = 1.0
    lambda_patch: float = 0.1
    lambda_adv: float = 0.0
    lower_tail_ratio: float = 0.2
    agg_alpha: float = 0.5
    agg_beta: float = 0.0
    use_tf32: bool = True


def ensure_project_dirs(cfg: ExperimentConfig) -> None:
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

