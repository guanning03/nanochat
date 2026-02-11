# D24 模型训练结果汇总

## 模型配置
- **模型深度**: 24 layers
- **参数量**: ~1.38B
- **训练设备**: 8x NVIDIA H100 80GB

---

## 1. BASE Pre-training (预训练)

### 训练信息
- **训练步数**: 27,000 steps
- **Checkpoint**: `d24_pretrain` (step 27000)
- **训练时间**: ~6.2 hours
- **Validation BPB**: **0.75431**

### 评估指标
- **CORE metric**: **0.2542**
- **ARC-Easy**: 0.5595
- **ARC-Challenge**: 0.1695
- **HellaSwag**: 0.3517
- **其他任务指标**: 见完整报告

---

## 2. MID Training (中间训练)

### 训练信息
- **训练步数**: 847 steps
- **Checkpoint**: `d24_mid` (step 847)
- **训练时间**: ~9 minutes
- **Minimum Validation BPB**: **0.3417**
- **Eval Every**: 150 steps
- **Save Every**: 150 steps (每次 eval 时保存)

### 评估指标

| 任务 | MID 结果 | 变化 (vs BASE) |
|------|---------|----------------|
| **ARC-Easy** | **0.5248** | -6.2% |
| **ARC-Challenge** | **0.4078** | +140.6% |
| **MMLU** | **0.3652** | - |
| **GSM8K** | **0.0159** | - |
| **HumanEval** | **0.0732** | - |
| **SpellingBee** | **0.9961** | - |
| **ChatCORE** | **0.3026** | - |

---

## 3. SFT (Supervised Fine-Tuning)

### 训练信息
- **训练步数**: 701 steps
- **Checkpoint**: `d24_sft` (step 701)
- **训练数据**: 22,443 rows
- **Training Loss**: 0.9117
- **Validation Loss**: 0.8456
- **Eval Every**: 100 steps
- **Save Every**: 100 steps (每次 eval 时保存)

### 评估指标

| 任务 | SFT 结果 | MID 结果 | 变化 | Karpathy 参考 (d20) |
|------|---------|---------|------|---------------------|
| **ARC-Easy** | **0.5547** | 0.5248 | +2.99% | 0.3876 |
| **ARC-Challenge** | **0.4215** | 0.4078 | +1.37% | 0.2807 |
| **MMLU** | **0.3644** | 0.3652 | -0.08% | 0.3151 |
| **GSM8K** | **0.1274** | 0.0159 | +701% | 0.0455 |
| **HumanEval** | **0.0244** | 0.0732 | -66.7% | 0.0854 |
| **SpellingBee** | **1.0000** | 0.9961 | +0.39% | - |
| **ChatCORE** | **0.3232** | 0.3026 | +6.8% | 0.0884 |

---

## 总结对比表

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2542   | -        | -        | -        |
| Validation BPB  | 0.75431  | 0.3417   | -        | -        |
| ARC-Challenge   | -        | 0.4078   | 0.4215   | -        |
| ARC-Easy        | -        | 0.5248   | 0.5547   | -        |
| GSM8K           | -        | 0.0159   | 0.1274   | -        |
| HumanEval       | -        | 0.0732   | 0.0244   | -        |
| MMLU            | -        | 0.3652   | 0.3644   | -        |
| ChatCORE        | -        | 0.3026   | 0.3232   | -        |

---

## 关键发现

### ✅ 表现优秀的任务
1. **GSM8K**: SFT 后从 0.0159 → 0.1274，提升 **701%**
2. **ARC-Easy**: SFT 后达到 **0.5547**，优于 Karpathy 的 d20 模型 (0.3876)
3. **ARC-Challenge**: SFT 后达到 **0.4215**，优于 Karpathy 的 d20 模型 (0.2807)
4. **ChatCORE**: SFT 后达到 **0.3232**，优于 Karpathy 的 d20 模型 (0.0884)

### 📊 与 Karpathy d20 模型对比
- **ARC-Easy**: 0.5547 vs 0.3876 (**+43.2%**)
- **ARC-Challenge**: 0.4215 vs 0.2807 (**+50.2%**)
- **MMLU**: 0.3644 vs 0.3151 (**+15.6%**)
- **GSM8K**: 0.1274 vs 0.0455 (**+180%**)
- **ChatCORE**: 0.3232 vs 0.0884 (**+265%**)

### ⚠️ 需要注意
- **HumanEval**: SFT 后从 0.0732 降至 0.0244，可能需要进一步优化

---

## Checkpoint 位置

- **BASE**: `/home/azanette/.cache/nanochat/base_checkpoints/d24_pretrain/`
- **MID**: `/home/azanette/.cache/nanochat/mid_checkpoints/d24_mid/`
- **SFT**: `/home/azanette/.cache/nanochat/chatsft_checkpoints/d24_sft/`

---

## 下一步建议

1. **RL Training**: 可以运行 `train_rl_d24.sh` 进一步提升 GSM8K 性能
2. **HumanEval 优化**: 考虑调整 SFT 数据混合或训练参数
3. **完整评估**: 运行完整的 CORE 评估以获取更全面的性能指标

---

*生成时间: 2026-02-09*
*报告来源: /home/azanette/.cache/nanochat/report/report.md*
