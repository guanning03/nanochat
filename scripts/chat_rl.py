"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Advantage estimator can be selected via --adv-estimator:
   - grpo: (reward - mu) / std
   - rloo: reward - mu (default)
   - maxrl: (reward - mu) / mu

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import math
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="sft", help="mid|sft - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from (for loading checkpoint)")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-model-tag", type=str, default=None, help="model tag for saving RL checkpoint (defaults to model-tag if not set)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--eval-num-samples", type=int, default=20, help="number of samples per problem during evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Advantage estimation
parser.add_argument("--adv-estimator", type=str, default="rloo", choices=["grpo", "rloo", "maxrl"], 
                     help="advantage estimator: grpo=(r-mu)/std, rloo=r-mu, maxrl=(r-mu)/mu")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
# Use WANDB_PROJECT environment variable if set, otherwise default to "nanochat-rl"
wandb_project = os.getenv("WANDB_PROJECT", "nanochat-rl")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=wandb_project, name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    batch_counter = 0  # counter to track batch number for seed generation
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((batch_counter, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed, # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages based on the selected estimator
        mu = rewards.mean()
        if args.adv_estimator == "grpo":
            # GRPO: (reward - mu) / std
            std = rewards.std(unbiased=False)  # use population std for consistency
            advantages = (rewards - mu) / (std + 1e-8)  # add small epsilon to avoid division by zero
        elif args.adv_estimator == "rloo":
            # RLOO: reward - mu
            advantages = rewards - mu
        elif args.adv_estimator == "maxrl":
            # MaxRL: (reward - mu) / mu
            # Note: mu is non-negative (rewards are 0 or 1), so abs() is redundant but harmless
            advantages = (rewards - mu) / (mu + 1e-8)  # add small epsilon to avoid division by zero
        else:
            raise ValueError(f"Unknown adv_estimator: {args.adv_estimator}")
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        batch_counter += 1  # increment counter for next batch
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Precomputed combination table for efficient lookup
# Global variable to store log C(n, k) values: comb_table[(n, k)] = log C(n, k)
comb_table = None

def init_comb_table(max_n, device):
    """
    Precompute and store log C(n, k) for all n <= max_n and k <= n.
    Stores values in a dictionary for O(1) lookup.
    """
    global comb_table
    if comb_table is not None:
        print0(f"组合数表已初始化，跳过重复初始化")
        return  # Already initialized
    
    print0(f"开始预计算组合数表：n <= {max_n}，设备: {device}")
    print0(f"这将计算所有 C(n, k) 的对数值，用于加速 pass@k 计算")
    comb_table = {}
    total_entries = 0
    
    for n in range(max_n + 1):
        for k in range(n + 1):
            if k > n or k < 0:
                value = torch.tensor(float('-inf'), device=device)
            elif k == 0 or k == n:
                # C(n, 0) = C(n, n) = 1, log(1) = 0
                value = torch.tensor(0.0, device=device)
            else:
                # Use the more stable formula: log C(n, k) = sum(log(i)) for i from (n-k+1) to n - sum(log(i)) for i from 1 to k
                numerator_range = torch.arange(n - k + 1, n + 1, dtype=torch.float32, device=device)
                denominator_range = torch.arange(1, k + 1, dtype=torch.float32, device=device)
                log_numerator = torch.sum(torch.log(numerator_range))
                log_denominator = torch.sum(torch.log(denominator_range))
                value = log_numerator - log_denominator
            comb_table[(n, k)] = value
            total_entries += 1
    
    print0(f"✓ 组合数表初始化完成！共预计算 {len(comb_table)} 个条目 (C(n,k) 其中 n<= {max_n})")

def log_comb(n, k):
    """
    Compute log(C(n, k)) = log(n! / (k! * (n-k)!))
    Uses precomputed table if available, otherwise computes on the fly.
    Returns a torch tensor scalar.
    """
    global comb_table
    
    # Use precomputed table if available and value is in table
    if comb_table is not None and (n, k) in comb_table:
        return comb_table[(n, k)]
    
    # Fallback: compute on the fly if table not initialized or value not in table
    # Get device from table if available, otherwise use None (CPU)
    if comb_table is not None and (n, k) not in comb_table:
        print0(f"警告：组合数 C({n}, {k}) 不在预计算表中，将实时计算（这不应该发生，请检查 n_samples 配置）")
    device = comb_table[(0, 0)].device if comb_table is not None else None
    
    if k > n or k < 0:
        return torch.tensor(float('-inf'), device=device) if device else torch.tensor(float('-inf'))
    if k == 0 or k == n:
        return torch.tensor(0.0, device=device) if device else torch.tensor(0.0)
    # Use the more stable formula: log C(n, k) = sum(log(i)) for i from (n-k+1) to n - sum(log(i)) for i from 1 to k
    if device:
        numerator_range = torch.arange(n - k + 1, n + 1, dtype=torch.float32, device=device)
        denominator_range = torch.arange(1, k + 1, dtype=torch.float32, device=device)
    else:
        numerator_range = torch.arange(n - k + 1, n + 1, dtype=torch.float32)
        denominator_range = torch.arange(1, k + 1, dtype=torch.float32)
    log_numerator = torch.sum(torch.log(numerator_range))
    log_denominator = torch.sum(torch.log(denominator_range))
    return log_numerator - log_denominator

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate num_samples samples, handling the case where num_samples > device_batch_size
        generated_token_sequences = []
        masks = []
        remaining_samples = num_samples
        while remaining_samples > 0:
            batch_size = min(remaining_samples, args.device_batch_size)
            generated_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=batch_size,
                max_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k
            )
            generated_token_sequences.extend(generated_batch)
            masks.extend(masks_batch)
            remaining_samples -= batch_size
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        print0(f"\n{'='*60}")
        print0(f"开始评估模型性能 (Step {step}/{num_steps})")
        print0(f"{'='*60}")
        model.eval()
        n_samples = args.eval_num_samples  # total number of samples per problem during evaluation
        print0(f"评估配置：每个问题生成 {n_samples} 个答案样本，评估 {args.eval_examples} 个问题")
        
        # Initialize combination table on first evaluation
        if comb_table is None:
            print0(f"首次评估，初始化组合数表...")
            init_comb_table(max_n=n_samples, device=device)
        else:
            print0(f"使用已初始化的组合数表（共 {len(comb_table)} 个条目）")
        
        # Compute k values: 2^x where x <= log2(n_samples), i.e., [1, 2, 4, 8, ...]
        max_exp = int(math.log2(n_samples))
        k_values = [2**x for x in range(max_exp + 1)]  # [1, 2, 4, 8, ...] up to n_samples
        print0(f"将计算 pass@k，其中 k = {k_values} (2的幂次方，最大到 {n_samples})")
        
        # Use dictionary to store pass@k values for the selected k values
        passk_dict = {k: torch.tensor(0.0, device=device) for k in k_values}
        
        print0(f"开始生成答案样本并评估...")
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=n_samples, max_examples=args.eval_examples, temperature=1.0)
            records = list(records_iter) # collect all records
        
        print0(f"✓ 答案生成完成，共收集到 {len(records)} 个问题的评估结果")
        print0(f"开始计算 pass@k（使用组合数公式：pass@k = 1 - C(n-c, k) / C(n, k)）")
        
        # Compute pass@k using combination formula: pass@k = 1 - C(n-c, k) / C(n, k)
        # where n = total samples, c = number of correct answers
        processed_count = 0
        for record in records:
            outcomes = record["outcomes"]
            n = len(outcomes)
            c = sum(int(o["is_correct"]) for o in outcomes)  # number of correct answers (explicit int conversion)
            
            # Compute pass@k only for k values in k_values (powers of 2)
            for k in k_values:
                if k > n:
                    # If k > n, we can't select k samples from n available
                    # If c > 0, we already have correct answers, so pass@k = 1
                    # If c == 0, no correct answers, so pass@k = 0
                    passk_dict[k] += 1.0 if c > 0 else 0.0
                elif c == 0:
                    # No correct answers, pass@k = 0
                    passk_dict[k] += 0.0
                elif k > n - c:
                    # If k > (n - c), we must select at least one correct answer
                    passk_dict[k] += 1.0
                else:
                    # Standard formula: pass@k = 1 - C(n-c, k) / C(n, k)
                    # Using logarithms: pass@k = 1 - exp(log_C(n-c, k) - log_C(n, k))
                    # Note: In this branch, we have k <= n - c, so C(n-c, k) is valid (not zero)
                    # Values from comb_table are already on the correct device
                    log_comb_wrong = log_comb(n - c, k)
                    log_comb_total = log_comb(n, k)
                    log_ratio = log_comb_wrong - log_comb_total
                    passk_dict[k] += 1.0 - torch.exp(log_ratio)
            
            processed_count += 1
            if processed_count % 50 == 0:
                print0(f"  已处理 {processed_count}/{len(records)} 个问题...")
        
        print0(f"✓ 完成所有问题的 pass@k 计算")
        
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            print0(f"分布式训练：聚合所有 {ddp_world_size} 个 rank 的评估结果...")
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            for k in k_values:
                dist.all_reduce(passk_dict[k], op=dist.ReduceOp.SUM)
            print0(f"✓ 聚合完成，总问题数: {num_records.item()}")
        
        # Normalize by the total number of records
        print0(f"归一化 pass@k 结果（除以总问题数 {num_records.item()}）...")
        for k in k_values:
            passk_dict[k] = passk_dict[k] / num_records.item()
        
        # Format output
        print_passk = [f"Pass@{k}: {passk_dict[k].item():.4f}" for k in k_values]
        print0(f"\n{'='*60}")
        print0(f"Step {step} 评估结果:")
        print0(f"  {', '.join(print_passk)}")
        print0(f"{'='*60}\n")
        
        log_passk = {f"pass@{k}": passk_dict[k].item() for k in k_values}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        # Use output_model_tag if specified, otherwise use model_tag, otherwise default to d{depth}
        output_dirname = args.output_model_tag if args.output_model_tag else (args.model_tag if args.model_tag else f"d{depth}")
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI args
])

wandb_run.finish() # wandb run finish
compute_cleanup()
