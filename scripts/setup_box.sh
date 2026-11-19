#!/usr/bin/env bash
# Box setup for the text-to-weight-update proof of concept. Verbose on purpose.
set -euo pipefail
ROOT=/workspace/poc
mkdir -p $ROOT/{models,data,teachers,baselines,meta,evals,logs,audit}
exec > >(tee -a $ROOT/logs/setup.log) 2>&1
echo "[setup] start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace | tail -1
python -c "import torch,sys;print('[setup] torch',torch.__version__,'cuda',torch.version.cuda,'py',sys.version.split()[0])"

echo "[setup] apt tools"
apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux git curl build-essential >/dev/null  # gcc: Triton (flash-linear-attention) JIT needs a C compiler

echo "[setup] python packages (uv, nothing newer than 2026-11-17 = 24h release-age delay)"
export UV_CACHE_DIR=$ROOT/hf_cache/uv HF_HOME=$ROOT/hf_cache
pip install -q uv
EXCL=2026-11-17T00:00:00Z
uv pip install --system -q --exclude-newer $EXCL \
  "transformers>=4.57" peft accelerate safetensors datasets "huggingface_hub[hf_transfer]" \
  "flash-linear-attention<0.5" einops sentencepiece pyyaml scipy matplotlib tqdm "brotli>=1.2"  # httpx2 needs Brotli>=1.2; fla 0.6 needs a newer triton than torch 2.8 ships, 0.4.2 works
python - <<'EOF'
import transformers, torch
print("[setup] transformers", transformers.__version__)
try:
    import fla; print("[setup] flash-linear-attention", getattr(fla, "__version__", "?"))
except Exception as e:
    print("[setup] WARN fla import failed:", e)
EOF

echo "[setup] packages done $(date -u +%FT%TZ)"
