#!/bin/bash

# MID (Midtraining) 训练脚本
# 从 BASE checkpoint 加载，学习对话格式、工具使用、多选题等
# 数据集: SmolTalk (460K) + MMLU (100K) + GSM8K (8K) + identity (2K) + Spelling (280K) = 848K samples

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
    WANDB_RUN="d24_mid_${CURRENT_TIME}"
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
# MID 训练配置
# 从 BASE checkpoint 加载

BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_pretrain}"  # BASE checkpoint 的 tag
BASE_MODEL_STEP="${BASE_MODEL_STEP:-}"            # 如果为空，使用最新的 checkpoint

echo "=========================================="
echo "MID (Midtraining) 训练"
echo "=========================================="
echo "配置:"
echo "  - Source: BASE checkpoint"
echo "  - Model Tag: $BASE_MODEL_TAG"
if [ -n "$BASE_MODEL_STEP" ]; then
    echo "  - Model Step: $BASE_MODEL_STEP"
else
    echo "  - Model Step: 最新 checkpoint"
fi
echo "  - Device Batch Size: 16 (d24 模型，使用较小 batch size)"
echo "  - Max Seq Len: 2048"
echo "  - Total Batch Size: 524288 tokens"
echo "  - Eval Every: 150 steps"
echo "  - Save Every: 150 steps (每次 eval 时保存 checkpoint)"
echo "  - Output Model Tag: d24_mid (checkpoint 保存路径)"
echo "  - 数据集:"
echo "    * SmolTalk: 460K (通用对话)"
echo "    * MMLU: 100K (多选题)"
echo "    * GSM8K: 8K (数学问题)"
echo "    * Identity: 2K (身份对话，2 epochs)"
echo "    * SimpleSpelling: 200K (简单拼写)"
echo "    * SpellingBee: 80K (拼写游戏)"
echo "  - 总计: ~848K samples"
echo "  - wandb run: $WANDB_RUN"
echo ""

# 构建训练命令
MID_CMD="torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --run=$WANDB_RUN \
    --model-tag=$BASE_MODEL_TAG \
    --output-model-tag=d24_mid \
    --device-batch-size=16 \
    --max-seq-len=2048 \
    --total-batch-size=524288 \
    --eval-every=150 \
    --save-every=150"

if [ -n "$BASE_MODEL_STEP" ]; then
    MID_CMD="$MID_CMD --model-step=$BASE_MODEL_STEP"
fi

echo "执行命令:"
echo "$MID_CMD"
echo ""

# 运行 MID 训练
eval $MID_CMD

echo ""
echo "✓ MID 训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 模型评估
echo "=========================================="
echo "评估 MID 模型性能"
echo "=========================================="

echo "正在评估 MID 模型..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
    -i mid \
    --batch-size=16 \
    --model-tag=d24_mid

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
echo "MID 训练完成！"
echo "=========================================="
echo ""
echo "MID checkpoint 位置: $NANOCHAT_BASE_DIR/mid_checkpoints/d24_mid/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "下一步: 可以运行 SFT 训练"
echo "  使用脚本: train_sft_d24.sh"
echo "  注意: SFT 默认从 MID checkpoint (d24_mid) 加载"
echo ""
