#!/bin/bash

# SFT (Supervised Fine-Tuning) 训练脚本
# 从 MID checkpoint 加载，在多任务混合数据上训练
# 数据集: ARC-Easy, ARC-Challenge, GSM8K, SmolTalk, identity_conversations, Spelling (~23K samples)

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
    WANDB_RUN="d24_sft_${CURRENT_TIME}"
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
# 下载 identity_conversations 数据（如果不存在）
IDENTITY_CONV_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_CONV_FILE" ]; then
    echo "=========================================="
    echo "下载 identity_conversations 数据"
    echo "=========================================="
    curl -L -o "$IDENTITY_CONV_FILE" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    echo "✓ 下载完成"
    echo ""
fi

# -----------------------------------------------------------------------------
# SFT 训练配置
# 标准流程: BASE -> MID -> SFT -> RL
# 默认从 MID checkpoint 加载（推荐）
# 如果还没有 MID checkpoint，可以从 BASE 加载（设置 SOURCE_MODEL_TYPE=base）

SOURCE_MODEL_TYPE="${SOURCE_MODEL_TYPE:-mid}"  # mid（推荐）或 base
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-}"        # 如果为空，使用默认的 checkpoint
SOURCE_MODEL_STEP="${SOURCE_MODEL_STEP:-}"      # 如果为空，使用最新的 checkpoint

# 如果从 MID 加载，默认使用 d24_mid
if [ "$SOURCE_MODEL_TYPE" = "mid" ] && [ -z "$SOURCE_MODEL_TAG" ]; then
    SOURCE_MODEL_TAG="d24_mid"
    echo "提示: 从 MID checkpoint 加载，使用默认 tag: $SOURCE_MODEL_TAG"
    echo "      如果没有指定 step，将使用最新的 checkpoint (step 847)"
    echo ""
fi

# 如果从 BASE 加载，默认使用 d24_pretrain
if [ "$SOURCE_MODEL_TYPE" = "base" ] && [ -z "$SOURCE_MODEL_TAG" ]; then
    SOURCE_MODEL_TAG="d24_pretrain"
    echo "提示: 从 BASE checkpoint 加载，使用默认 tag: $SOURCE_MODEL_TAG"
    echo "      如果没有指定 step，将使用最新的 checkpoint"
    echo ""
fi

echo "=========================================="
echo "SFT (Supervised Fine-Tuning) 训练"
echo "=========================================="
echo "配置:"
echo "  - Source: $SOURCE_MODEL_TYPE"
if [ -n "$SOURCE_MODEL_TAG" ]; then
    echo "  - Model Tag: $SOURCE_MODEL_TAG"
fi
if [ -n "$SOURCE_MODEL_STEP" ]; then
    echo "  - Model Step: $SOURCE_MODEL_STEP"
fi
echo "  - Device Batch Size: 4"
echo "  - Num Epochs: 1"
echo "  - Eval Every: 100 steps"
echo "  - Save Every: 100 steps (每次 eval 时保存 checkpoint)"
echo "  - Output Model Tag: d24_sft (checkpoint 保存路径)"
echo "  - 数据集: ARC-Easy (2.3K) + ARC-Challenge (1.1K) + GSM8K (8K) + SmolTalk (10K) + identity (1K) + Spelling (600)"
echo "  - 总计: ~23K samples"
echo "  - wandb run: $WANDB_RUN"
echo ""

# 构建训练命令
# --model-tag 用于加载 MID checkpoint
# --output-model-tag 用于保存 SFT checkpoint 到指定目录
SFT_CMD="torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --source=$SOURCE_MODEL_TYPE \
    --run=$WANDB_RUN \
    --device-batch-size=4 \
    --num-epochs=1 \
    --eval-every=100 \
    --save-every=100 \
    --output-model-tag=d24_sft"

# 如果指定了 SOURCE_MODEL_TAG，用于加载 MID checkpoint
if [ -n "$SOURCE_MODEL_TAG" ]; then
    SFT_CMD="$SFT_CMD --model-tag=$SOURCE_MODEL_TAG"
fi

if [ -n "$SOURCE_MODEL_STEP" ]; then
    SFT_CMD="$SFT_CMD --model-step=$SOURCE_MODEL_STEP"
fi

echo "执行命令:"
echo "$SFT_CMD"
echo ""

# 运行 SFT 训练
eval $SFT_CMD

echo ""
echo "✓ SFT 训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 模型评估
echo "=========================================="
echo "评估 SFT 模型性能"
echo "=========================================="

echo "正在评估 SFT 模型..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
    -i sft \
    --batch-size=16

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
echo "SFT 训练完成！"
echo "=========================================="
echo ""
echo "SFT checkpoint 位置: $NANOCHAT_BASE_DIR/chatsft_checkpoints/d24_sft/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "下一步: 可以运行 RL 训练来进一步提升 GSM8K 性能"
echo "  使用脚本: train_rl_d24.sh"
echo "  注意: RL 默认从 SFT checkpoint (d24_sft) 加载"
echo ""
