#!/bin/bash

# 标准 d24 模型预训练脚本
# 适用于 8xH100 GPU 节点
# 使用方法: bash train_d24.sh

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
# 设置 wandb project name
export WANDB_PROJECT="nanochat_d24_guanning"
WANDB_RUN="pretrain"

# 显示 wandb 配置信息
echo "=========================================="
echo "Wandb 配置"
echo "=========================================="
echo "  - Entity: $WANDB_ENTITY"
echo "  - Project: $WANDB_PROJECT"
echo "  - Run Name: $WANDB_RUN"
echo "  - 状态: Wandb 记录已启用"
echo ""

# -----------------------------------------------------------------------------
# 3. 数据下载和 Tokenizer 训练

echo "=========================================="
echo "步骤 3/5: 下载数据并训练 Tokenizer"
echo "=========================================="

# 初始化报告目录
python -m nanochat.report reset

# 下载前 8 个数据 shards (~2B 字符) 用于训练 tokenizer
# 每个 shard 约 250M 字符，约 100MB (压缩后)
echo "正在下载前 8 个数据 shards (用于 tokenizer 训练)..."
python -m nanochat.dataset -n 8

# 后台下载更多数据 shards (约 370 个 shards，用于预训练)
# 总共需要约 10B tokens 的数据
# 注意: 实际需要的 shards 数量取决于你的训练配置
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
# 4. 等待数据下载完成

echo "=========================================="
echo "步骤 4/5: 等待数据下载完成"
echo "=========================================="

echo "等待数据下载完成 (PID: $DATASET_DOWNLOAD_PID)..."
wait $DATASET_DOWNLOAD_PID
echo "✓ 数据下载完成"
echo ""

# -----------------------------------------------------------------------------
# 5. d24 模型预训练

echo "=========================================="
echo "步骤 5/5: d24 模型预训练"
echo "=========================================="

echo "开始训练 d24 模型..."
echo "配置:"
echo "  - depth: 24"
echo "  - target-param-data-ratio: 10.5 (compute optimal)"
echo "  - device-batch-size: 32 (如果 OOM，会自动降级到 16)"
echo "  - fp8: 启用 (H100 支持)"
echo "  - wandb run: $WANDB_RUN"
echo ""

# d24 预训练命令
# 使用 compute optimal 的 target-param-data-ratio=10.5 (默认值)
# 如果 GPU 内存不足，device-batch-size 会自动通过梯度累积调整
# 评估优化参数：
#   - sample-every=-1: 关闭定期采样，节省时间
#   - save-every=-1: 只在最后保存 checkpoint，节省时间和磁盘
#   - core-metric-every=999999: 只在最后评估 CORE metric（评估很耗时）
#   - core-metric-max-per-task=-1: 运行完整的 CORE 评估
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=10.5 \
    --device-batch-size=32 \
    --fp8 \
    --run=$WANDB_RUN \
    --model-tag="d24" \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1

echo ""
echo "✓ 预训练完成！"
echo ""

# -----------------------------------------------------------------------------
# 6. 模型评估

echo "=========================================="
echo "评估模型性能"
echo "=========================================="

echo "正在评估模型 (CORE metric, BPB, samples)..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=32

echo ""
echo "✓ 评估完成！"
echo ""

# -----------------------------------------------------------------------------
# 7. 生成报告

echo "=========================================="
echo "生成训练报告"
echo "=========================================="

python -m nanochat.report generate

echo ""
echo "=========================================="
echo "训练流程全部完成！"
echo "=========================================="
echo ""
echo "模型 checkpoint 位置: $NANOCHAT_BASE_DIR/models/d24/"
echo "训练报告: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "下一步:"
echo "  1. 查看训练报告: cat $NANOCHAT_BASE_DIR/report/report.md"
echo "  2. 如需继续 SFT，运行: bash train_d24_sft.sh"
echo "  3. 或在 Python 中加载模型进行推理"
echo ""
