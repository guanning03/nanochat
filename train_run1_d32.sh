#!/bin/bash

# Run 1 配置的 d32 预训练脚本
# 基于 Run 1 (d24) 的配置，只改变 depth=32
# 用于研究不同深度模型的性能对比
# Commit: 348fbb3 (Jan 29 2026)

source /home/azanette/.bashrc
source /home/azanette/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
export WANDB_ENTITY=paprika_online

# 设置环境变量
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 生成时间戳（格式: YYYYMMDD_HHMMSS）
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# -----------------------------------------------------------------------------
# Wandb 配置
export WANDB_PROJECT="nanochat_d24_guanning"
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="d32_pretrain_${CURRENT_TIME}"
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
# 如果 tokenizer 已存在，跳过训练以保持一致性
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -f "$TOKENIZER_DIR/tokenizer.pkl" ] && [ -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Tokenizer 已存在，跳过训练 (使用: $TOKENIZER_DIR)"
    echo "如需重新训练，请删除该目录: rm -rf $TOKENIZER_DIR"
else
    echo "正在训练 tokenizer (vocab_size=32768)..."
    python -m scripts.tok_train
fi

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
# Run 1 配置的 d32 预训练
# 配置: depth=32, target-param-data-ratio=40, device-batch-size=8
# core-metric-every=3000 (每 3000 步评估一次 CORE)
# 注意: ratio=24 表示训练步数翻倍，device-batch-size=8 是 d32 的特性

echo "=========================================="
echo "步骤 3/3: Run 1 配置的 d32 预训练"
echo "=========================================="

echo "开始训练 d32 模型 (Run 1 配置，训练步数翻倍)..."
echo "配置:"
echo "  - depth: 32"
echo "  - target-param-data-ratio: 24 (训练步数翻倍)"
echo "  - device-batch-size: 8 (d32 模型更大，使用较小 batch size)"
echo "  - fp8: 未启用 (使用 bf16)"
echo "  - core-metric-every: 3000 (每 3000 步评估)"
echo "  - save-every: 3000 (每 3000 步保存 checkpoint)"
echo "  - wandb run: $WANDB_RUN"
echo ""

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 \
    --target-param-data-ratio=40 \
    --device-batch-size=8 \
    --run=$WANDB_RUN \
    --model-tag="d32_pretrain_${CURRENT_TIME}" \
    --sample-every=-1 \
    --save-every=3000 \
    --core-metric-every=3000 \
    --core-metric-max-per-task=-1

echo ""
echo "✓ Run 1 配置的 d32 预训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 模型评估

echo "=========================================="
echo "评估模型性能"
echo "=========================================="

echo "正在评估模型 (CORE metric, BPB, samples)..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=8

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
echo "Run 1 配置的 d32 预训练完成！"
echo "=========================================="
echo ""
echo "预期结果 (训练步数翻倍):"
echo "  - CORE score: 预期会有所提升（更多训练数据）"
echo "  - Validation BPB: 预期会降低（更多训练）"
echo "  - Total training time: 约翻倍（d32 模型更大，训练时间更长）"
echo "  - Steps: 约翻倍（取决于 target-param-data-ratio=40）"
echo ""
echo "模型 checkpoint 位置: $NANOCHAT_BASE_DIR/base_checkpoints/d32_pretrain_${CURRENT_TIME}/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "对比说明:"
echo "  - 此脚本与 train_run1_d24.sh 的主要区别："
echo "    * depth=32 (vs d24)"
echo "    * device-batch-size=8 (vs 16，因为 d32 模型更大)"
echo "  - 其他参数保持一致（target-param-data-ratio=40，训练步数翻倍）"
echo "  - 可用于研究不同深度模型的性能差异"
echo ""
