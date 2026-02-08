#!/bin/bash

# 复现 Run 2 (d26) 预训练脚本
# 对应 LEADERBOARD.md 中的 Run 2
# 预期结果: CORE 0.2578, val_bpb 0.74504, time ~2.91 hours
# Commit: a67eba3 (Feb 2 2026)
# 关键改进: 启用 FP8 训练

source /home/azanette/.bashrc
source /home/azanette/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
export WANDB_ENTITY=paprika_online

# 设置环境变量
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Wandb 配置
export WANDB_PROJECT="nanochat_d24_guanning"
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="run2_d26_reproduce"
fi

echo "=========================================="
echo "Wandb 配置"
echo "=========================================="
echo "  - Entity: $WANDB_ENTITY"
echo "  - Project: $WANDB_PROJECT"
echo "  - Run Name: $WANDB_RUN"
echo "  - 状态: Wandb 记录已启用"
echo ""

# -----------------------------------------------------------------------------
# 数据下载和 Tokenizer 训练

echo "=========================================="
echo "步骤 1/3: 下载数据并训练 Tokenizer"
echo "=========================================="

python -m nanochat.report reset

# 下载前 8 个数据 shards (~2B 字符) 用于训练 tokenizer
echo "正在下载前 8 个数据 shards (用于 tokenizer 训练)..."
python -m nanochat.dataset -n 8

# 后台下载更多数据 shards (约 370 个 shards，用于预训练)
echo "正在后台下载预训练数据 (370 shards)..."
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!

# 训练 tokenizer (vocab size = 32768)
echo "正在训练 tokenizer (vocab_size=32768)..."
python -m scripts.tok_train

# 评估 tokenizer
echo "正在评估 tokenizer..."
python -m scripts.tok_eval

echo "✓ Tokenizer 训练完成"
echo ""

# -----------------------------------------------------------------------------
# 等待数据下载完成

echo "=========================================="
echo "步骤 2/3: 等待数据下载完成"
echo "=========================================="

echo "等待数据下载完成 (PID: $DATASET_DOWNLOAD_PID)..."
wait $DATASET_DOWNLOAD_PID
echo "✓ 数据下载完成"
echo ""

# -----------------------------------------------------------------------------
# Run 2 (d26) 预训练
# 配置: depth=26, target-param-data-ratio=8.5, device-batch-size=16
# 关键: 启用 FP8 训练
# core-metric-every=999999 (只在最后评估 CORE)

echo "=========================================="
echo "步骤 3/3: Run 2 (d26) 预训练"
echo "=========================================="

echo "开始训练 d26 模型 (Run 2 配置)..."
echo "配置:"
echo "  - depth: 26"
echo "  - target-param-data-ratio: 8.5 (undertrained d26)"
echo "  - device-batch-size: 16"
echo "  - fp8: 启用 (关键改进)"
echo "  - core-metric-every: 999999 (只在最后评估)"
echo "  - wandb run: $WANDB_RUN"
echo ""

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.5 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN \
    --model-tag="run2_d26" \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1

echo ""
echo "✓ Run 2 (d26) 预训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 模型评估

echo "=========================================="
echo "评估模型性能"
echo "=========================================="

echo "正在评估模型 (CORE metric, BPB, samples)..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=16

echo ""
echo "✓ 评估完成！"
echo ""

# -----------------------------------------------------------------------------
# 生成报告

echo "=========================================="
echo "生成训练报告"
echo "=========================================="

python -m nanochat.report generate

echo ""
echo "=========================================="
echo "Run 2 (d26) 复现完成！"
echo "=========================================="
echo ""
echo "预期结果:"
echo "  - CORE score: ~0.2578 (应该 > 0.256525)"
echo "  - Validation BPB: ~0.74504"
echo "  - Total training time: ~10493 秒 (~2.91 小时)"
echo "  - Steps: ~14889"
echo ""
echo "关键改进: FP8 训练使速度提升 ~4.3%"
echo ""
echo "模型 checkpoint 位置: $NANOCHAT_BASE_DIR/models/run2_d26/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
