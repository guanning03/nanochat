# RL Training 参数验证与说明

## ✅ Karpathy 推荐设置验证

根据 `scripts/chat_rl.py` 的代码注释和实现，当前 RL 训练脚本使用的是 **Karpathy 的标准推荐设置**。

### 方法说明
代码注释明确说明（第 1-10 行）：
- 使用简化的 GRPO（类似 REINFORCE）
- **删除 trust region**（无 KL 正则化到参考模型）
- **严格 on-policy**（无需 PPO ratio+clip）
- 使用 DAPO 风格的 token-level 归一化
- 优势函数使用 `(r - mu)` 而非 `(r - mu)/sigma`

---

## 📊 训练参数详情

### 1. Batch Size 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **`device-batch-size`** | **8** | 每个 GPU 的前向传播批次大小 |
| **`examples-per-step`** | **16** | 每个优化步骤处理的训练样本数（跨所有 ranks） |
| **`num-samples`** | **16** | 每个问题生成的答案数量（rollout number） |

**总批次大小计算**：
- 每个 step 的总序列数 = `examples-per-step × num-samples` = 16 × 16 = **256 个序列**
- 8 GPUs 时，每个 GPU 处理 = 256 / 8 = **32 个序列**
- 每个 GPU 需要分 `32 / 8 = 4` 个 pass 处理（受 `device-batch-size=8` 限制）

### 2. Rollout Number（采样数量）

- **`num-samples=16`**: 每个 GSM8K 问题生成 **16 个答案**
- 用于计算 pass@k 评估（k=1 到 16）
- 这是标准的 on-policy 采样设置

### 3. Learning Rate 配置

| 参数类型 | 值 | 优化器 |
|---------|-----|--------|
| **`embedding-lr`** | **0.2** | Adam |
| **`unembedding-lr`** | **0.004** | Adam |
| **`matrix-lr`** | **0.02** | Muon |
| **`init-lr-frac`** | **0.05** | 初始 LR = base LR × 0.05 |

**学习率调度**：
- 线性衰减：`lr_multiplier = 1.0 - step / num_steps`
- 从初始 LR（base × 0.05）线性衰减到 0

### 4. On-Policy 验证

✅ **严格 on-policy**：
- 代码第 8 行明确说明：`"We are on policy, so there's no need for PPO ratio+clip"`
- 代码第 279 行注释：`"Note, there is no need to add PPO ratio+clip because we are on policy"`
- 每个训练步骤都使用当前策略模型生成新的 rollouts
- 没有使用旧策略的 importance sampling

### 5. Save 和 Eval 间隔

| 参数 | 值 | 说明 |
|------|-----|------|
| **`save-every`** | **60** | 每 60 步保存一次 checkpoint |
| **`eval-every`** | **60** | 每 60 步进行一次评估 |
| **`eval-examples`** | **400** | 评估时使用的测试样本数 |

**保存逻辑**：
- 跳过第 0 步
- 每 `save-every` 步保存一次
- 最后一步（`num_steps - 1`）也会保存

---

## 🎯 训练配置总结

### 当前脚本设置（`train_rl_d24.sh`）

```bash
--device-batch-size=8          # ✅ 默认值
--examples-per-step=16          # ✅ 默认值
--num-samples=16                # ✅ 默认值（rollout number）
--num-epochs=1                  # ✅ 默认值
--max-new-tokens=256            # ✅ 默认值
--temperature=1.0               # ✅ 默认值
--top-k=50                      # ✅ 默认值
--eval-every=60                 # ✅ 默认值
--eval-examples=400             # ✅ 默认值
--save-every=60                 # ✅ 默认值
--output-model-tag=d24_rl       # ✅ 新增：指定输出目录
```

### Learning Rate（使用默认值）

```bash
--embedding-lr=0.2              # ✅ 默认值
--unembedding-lr=0.004          # ✅ 默认值
--matrix-lr=0.02                # ✅ 默认值
--init-lr-frac=0.05             # ✅ 默认值
```

---

## ✅ 验证结论

1. **✅ 所有参数都是 Karpathy 的默认推荐值**
2. **✅ 严格 on-policy 训练**（无 PPO ratio+clip）
3. **✅ 使用简化的 GRPO 方法**（无 trust region）
4. **✅ Checkpoint 保存到 `d24_rl` 目录**（已配置）

---

## 📝 训练流程

1. **数据**: GSM8K train split
2. **方法**: On-policy REINFORCE（简化 GRPO）
3. **评估**: GSM8K test split，计算 pass@k (k=1..16)
4. **输出**: Checkpoint 保存到 `/home/azanette/.cache/nanochat/chatrl_checkpoints/d24_rl/`

---

*生成时间: 2026-02-09*
*参考: `scripts/chat_rl.py`*
