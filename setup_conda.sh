#!/bin/bash

# Conda 环境快速设置脚本
# 使用方法: bash setup_conda.sh [环境名称]
# 默认环境名称: nanochat

ENV_NAME="${1:-nanochat}"

echo "=========================================="
echo "设置 Conda 环境: $ENV_NAME"
echo "=========================================="

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 conda"
    exit 1
fi

# 创建 conda 环境 (如果不存在)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在，跳过创建"
    echo "激活环境..."
    conda activate "$ENV_NAME"
else
    echo "正在创建 conda 环境: $ENV_NAME (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
    conda activate "$ENV_NAME"
fi

# 检查 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 安装 PyTorch (CUDA 12.8)
echo ""
echo "正在安装 PyTorch 2.9.1 (CUDA 12.8)..."
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 验证 PyTorch 安装
echo ""
echo "验证 PyTorch 安装..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 安装其他依赖
echo ""
echo "正在安装其他依赖包..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "警告: 未找到 requirements.txt，使用手动安装..."
    pip install datasets>=4.0.0 \
        fastapi>=0.117.1 \
        ipykernel>=7.1.0 \
        kernels>=0.11.7 \
        matplotlib>=3.10.8 \
        psutil>=7.1.0 \
        python-dotenv>=1.2.1 \
        regex>=2025.9.1 \
        rustbpe>=0.1.0 \
        scipy>=1.15.3 \
        setuptools>=80.9.0 \
        tabulate>=0.9.0 \
        tiktoken>=0.11.0 \
        tokenizers>=0.22.0 \
        torchao==0.15.0 \
        transformers>=4.57.3 \
        uvicorn>=0.36.0 \
        wandb>=0.21.3 \
        zstandard>=0.25.0
fi

echo ""
echo "=========================================="
echo "✓ Conda 环境设置完成！"
echo "=========================================="
echo ""
echo "环境名称: $ENV_NAME"
echo ""
echo "使用方法:"
echo "  1. 激活环境: conda activate $ENV_NAME"
echo "  2. 运行训练: bash train_d24.sh"
echo ""
