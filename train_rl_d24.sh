#!/bin/bash

# RL (Reinforcement Learning) 训练脚本
# 从 SFT checkpoint 加载，在 GSM8K 上使用 GRPO 训练
# 方法: 简化的 REINFORCE (GRPO without trust region)

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
# RL 训练参数
# Advantage estimator: grpo, rloo, maxrl
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"  # 默认使用 grpo

# -----------------------------------------------------------------------------
# Wandb 配置
export WANDB_PROJECT="nanochat_rl_guanning"
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="d24_rl_${ADV_ESTIMATOR}_${CURRENT_TIME}"
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
# RL 训练配置
# 默认从 SFT checkpoint 加载，如果要从 MID 加载，设置 SOURCE_MODEL_TAG 和 SOURCE_MODEL_STEP

SOURCE_MODEL_TYPE="${SOURCE_MODEL_TYPE:-sft}"  # sft 或 mid
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-}"        # 如果为空，使用默认的 sft checkpoint
SOURCE_MODEL_STEP="${SOURCE_MODEL_STEP:-}"      # 如果为空，使用最新的 checkpoint

echo "=========================================="
echo "RL (Reinforcement Learning) 训练"
echo "=========================================="
echo "配置:"
echo "  - Source: $SOURCE_MODEL_TYPE"
if [ -n "$SOURCE_MODEL_TAG" ]; then
    echo "  - Model Tag: $SOURCE_MODEL_TAG"
fi
if [ -n "$SOURCE_MODEL_STEP" ]; then
    echo "  - Model Step: $SOURCE_MODEL_STEP"
fi
echo "  - Advantage Estimator: $ADV_ESTIMATOR (grpo|rloo|maxrl)"
echo "  - 数据集: GSM8K (数学问题)"
echo "  - 方法: GRPO (简化版 REINFORCE)"
echo "  - Device Batch Size: 8"
echo "  - Examples Per Step: 32"
echo "  - Num Samples: 16 (每个问题生成16个答案)"
echo "  - Num Epochs: 20"
echo "  - Max New Tokens: 256"
echo "  - Temperature: 1.0"
echo "  - wandb run: $WANDB_RUN"
echo ""

# 构建训练命令
RL_CMD="torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- \
    --source=$SOURCE_MODEL_TYPE \
    --run=$WANDB_RUN \
    --adv-estimator=$ADV_ESTIMATOR \
    --device-batch-size=8 \
    --examples-per-step=32 \
    --num-samples=16 \
    --num-epochs=20 \
    --max-new-tokens=256 \
    --temperature=1.0 \
    --top-k=50 \
    --eval-every=50 \
    --eval-examples=400 \
    --eval-num-samples=16 \
    --save-every=50 \
    --output-model-tag=d24_rl_${ADV_ESTIMATOR}_${CURRENT_TIME}"

if [ -n "$SOURCE_MODEL_TAG" ]; then
    RL_CMD="$RL_CMD --model-tag=$SOURCE_MODEL_TAG"
fi

if [ -n "$SOURCE_MODEL_STEP" ]; then
    RL_CMD="$RL_CMD --model-step=$SOURCE_MODEL_STEP"
fi

echo "执行命令:"
echo "$RL_CMD"
echo ""
echo "调试信息:"
echo "  - WANDB_PROJECT 环境变量: $WANDB_PROJECT"
echo ""

# 运行 RL 训练
eval $RL_CMD

echo ""
echo "✓ RL 训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 模型评估（仅评估 GSM8K）
echo "=========================================="
echo "评估 RL 模型性能 (GSM8K)"
echo "=========================================="

echo "正在评估 RL 模型在 GSM8K 上的表现..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
    -i rl \
    -a GSM8K \
    --num-samples=20

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
echo "RL 训练完成！"
echo "=========================================="
echo ""
echo "RL checkpoint 位置: $NANOCHAT_BASE_DIR/chatrl_checkpoints/d24_rl_${ADV_ESTIMATOR}_${CURRENT_TIME}/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "训练流程完成: BASE -> MID -> SFT -> RL"
echo ""
echo "提示: 可以通过设置环境变量 ADV_ESTIMATOR 来选择优势估计器"
echo "  例如: ADV_ESTIMATOR=rloo bash train_rl_d24.sh"
echo "  可选值: grpo (默认), rloo, maxrl"
echo ""