#!/usr/bin/env bash
# Run on remote pod: bash run_bench.sh
set -e

pip install einops uv --quiet

# Clone the repo
git clone https://github.com/KevinZhou-cs336/assignment2-systems.git /root/repo
cd /root/repo

# Install cs336-basics
pip install -e cs336-basics --quiet

# Run float32 benchmark
echo "=== float32 ===" > /root/attn_results_fp32.txt
python -m cs336_systems.pytorch_attention --dtype float32 --num-warmup 10 --num-steps 100 2>&1 | tee -a /root/attn_results_fp32.txt

# Run bfloat16 benchmark
echo "=== bfloat16 ===" > /root/attn_results_bf16.txt
python -m cs336_systems.pytorch_attention --dtype bfloat16 --num-warmup 10 --num-steps 100 2>&1 | tee -a /root/attn_results_bf16.txt

echo "Done"
