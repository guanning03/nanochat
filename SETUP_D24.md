# d24 模型预训练环境配置指南

## 快速开始

如果你有 8xH100 GPU 节点，最简单的方式是直接运行：

```bash
bash train_d24.sh
```

这个脚本会自动完成所有步骤。脚本会自动检测你是否在使用 conda 环境。

---

## 详细步骤说明

### 1. 环境要求

- **GPU**: 8xH100 (每个 80GB)
- **系统**: Linux
- **Python**: >=3.10
- **CUDA**: 12.8 (用于 PyTorch)

### 2. Python 环境配置

#### 方式 A: 使用 Conda (推荐，如果你习惯使用 conda)

**快速设置** (推荐):
```bash
# 运行自动设置脚本
bash setup_conda.sh

# 或者指定环境名称
bash setup_conda.sh my_nanochat_env
```

**手动设置**:
```bash
# 创建 conda 环境 (Python 3.10+)
conda create -n nanochat python=3.10 -y
conda activate nanochat

# 安装 PyTorch (CUDA 12.8)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖 (使用 requirements.txt)
pip install -r requirements.txt
```

#### 方式 B: 使用 uv (项目默认方式，更快)

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 创建虚拟环境
uv venv

# 安装依赖 (GPU 版本)
uv sync --extra gpu

# 激活虚拟环境
source .venv/bin/activate
```

**注意**: 
- `uv sync --extra gpu` 会自动安装所有依赖，包括 PyTorch 2.9.1 (CUDA 12.8)
- 如果使用 conda，脚本会自动检测 conda 环境并跳过 uv 安装步骤

### 3. 数据下载

nanochat 使用 HuggingFace FineWeb 数据集：

```bash
# 设置数据缓存目录
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 下载前 8 个 shards (用于 tokenizer 训练，约 800MB)
python -m nanochat.dataset -n 8

# 下载更多 shards (用于预训练，约 370 个 shards，约 37GB)
# 注意：这会下载约 10B tokens 的数据
python -m nanochat.dataset -n 370
```

**数据说明**:
- 每个 shard 约 250M 字符，压缩后约 100MB
- 总共约 1822 个 shards 可用
- d24 训练需要约 370 shards (约 10B tokens)

### 4. Tokenizer 训练

```bash
# 训练 tokenizer (vocab_size=32768)
python -m scripts.tok_train

# 评估 tokenizer
python -m scripts.tok_eval
```

### 5. d24 模型预训练

**标准配置 (compute optimal)**:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=10.5 \
    --device-batch-size=32 \
    --fp8 \
    --run="d24_training" \
    --model-tag="d24"
```

**参数说明**:
- `--depth=24`: 24 层 Transformer
- `--target-param-data-ratio=10.5`: compute optimal 的数据:参数比例
- `--device-batch-size=32`: 每个 GPU 的 batch size (如果 OOM，改为 16)
- `--fp8`: 启用 FP8 训练 (H100 支持，更快)
- `--run`: wandb run 名称
- `--model-tag`: checkpoint 保存标签

**如果 GPU 内存不足**:
- 将 `--device-batch-size` 改为 `16` 或 `8`
- 脚本会自动通过梯度累积保持总 batch size

### 6. 模型评估

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=32
```

这会评估：
- CORE metric (DCLM 评分)
- Validation bits per byte (BPB)
- 生成样本

### 7. Wandb 配置 (可选但推荐)

```bash
# 登录 wandb
wandb login

# 设置 run 名称
export WANDB_RUN="d24_$(date +%Y%m%d)"
```

如果不使用 wandb，脚本会使用 "dummy" 模式（不记录日志）。

---

## 完整训练流程示例

### 使用 Conda

```bash
# 1. 创建并激活 conda 环境
conda create -n nanochat python=3.10 -y
conda activate nanochat

# 2. 安装 PyTorch
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 使用 uv

```bash
# 1. 环境配置
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv venv
uv sync --extra gpu
source .venv/bin/activate
```

### 后续步骤 (两种方式相同)

# 2. 设置环境变量
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 3. 下载数据 (先下载 8 个用于 tokenizer)
python -m nanochat.dataset -n 8

# 4. 训练 tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

# 5. 下载更多数据 (后台下载，同时可以开始训练)
python -m nanochat.dataset -n 370 &

# 6. 预训练 d24
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=10.5 \
    --device-batch-size=32 \
    --fp8 \
    --run="d24" \
    --model-tag="d24"

# 7. 评估
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=32
```

---

## 常见问题

### Q: 如何使用 conda 环境？
A: 
1. 创建 conda 环境: `conda create -n nanochat python=3.10 -y`
2. 激活环境: `conda activate nanochat`
3. 安装 PyTorch: `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`
4. 安装依赖: `pip install -r requirements.txt`
5. 运行脚本: `bash train_d24.sh` (脚本会自动检测 conda 环境)

### Q: conda 和 uv 环境可以共存吗？
A: 可以。脚本会自动检测你是否在 conda 环境中。如果在 conda 环境中，会使用 conda；否则使用 uv。

### Q: 数据下载很慢怎么办？
A: 可以在后台下载数据，同时训练 tokenizer：
```bash
python -m nanochat.dataset -n 370 &
# 然后训练 tokenizer
python -m scripts.tok_train
# 等待数据下载完成
wait
```

### Q: 如何监控训练进度？
A: 
1. 使用 wandb: `wandb login` 后设置 `WANDB_RUN` 环境变量
2. 查看日志输出
3. 检查 checkpoint: `$NANOCHAT_BASE_DIR/models/d24/`

### Q: 训练需要多长时间？
A: 
- d24 compute optimal 训练: 约 2-3 小时 (8xH100)
- 数据下载: 取决于网络速度 (约 37GB)
- Tokenizer 训练: 约 10-20 分钟

### Q: 如何调整训练参数？
A: 查看 `scripts/base_train.py` 中的参数说明，或运行：
```bash
python -m scripts.base_train --help
```

---

## 参考

- [speedrun.sh](runs/speedrun.sh): GPT-2 级别模型的完整训练流程
- [miniseries.sh](runs/miniseries.sh): 多深度模型的批量训练
- [LEADERBOARD.md](dev/LEADERBOARD.md): 训练参数详细说明
