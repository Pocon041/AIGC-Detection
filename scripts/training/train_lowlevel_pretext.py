from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def main():
    raise SystemExit("??? Beyond Generation pretext ????? original?VAE reconstruction ? diffusion-denoised variants???? train_lowlevel_precomputed.py")


if __name__ == "__main__":
    main()


