#!/usr/bin/env bash
# tmux session "download": starts as soon as packages are in, in parallel with the data build.
set -euo pipefail
ROOT=/workspace/poc
exec > >(tee -a $ROOT/logs/download.log) 2>&1
until grep -q "\[setup\] packages done" $ROOT/logs/setup.log 2>/dev/null; do sleep 10; done
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=$ROOT/hf_cache
echo "[download] start $(date -u +%FT%TZ)"
# retry wrapper for transient network failures
dl() { for i in 1 2 3 4 5 6; do hf download "$@" && return 0; echo "[download] retry $i: $*"; sleep $((10*i)); done; return 1; }
if [ ! -d $ROOT/t2l ]; then git clone -q --depth 1 https://github.com/SakanaAI/text-to-lora.git $ROOT/t2l; fi
( cd $ROOT/t2l && echo "[download] t2l commit $(git rev-parse HEAD)" )
# tokenizer and configs first so the data build can start, then the weights
dl Qwen/Qwen3.5-2B --local-dir $ROOT/models/Qwen3.5-2B --exclude "*.safetensors"
echo "[download] tokenizer done $(date -u +%FT%TZ)"
dl Qwen/Qwen3.5-2B --local-dir $ROOT/models/Qwen3.5-2B
dl BAAI/bge-large-en-v1.5 --local-dir $ROOT/models/bge-large-en-v1.5 --include "*.json" --include "*.txt" --include "model.safetensors"
du -sh $ROOT/models/*
df -h /workspace | tail -1
echo "[download] done $(date -u +%FT%TZ)"
