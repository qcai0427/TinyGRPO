from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map=None,
):
    # 初始化 tokenizer (分词器)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    # 如果该模型没有专门针对 padding 的占位 token，就默认把结尾 token 的 ID 当做 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 初始化核心的因果语言模型 (Causal Language Model) 
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        # SDPA: PyTorch 原生的加速缩放点积注意力，能够大大降低显存消耗、提升推理与训练速度 (类似 Flash Attention)
        attn_implementation="sdpa",  
        # 如果 bf16 为 True 就在加载时利用 bfloat16 数据精度以节省显存 (LLM 主流训练策略)
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek 针对推理 RL 构建的一个典型的 system_prompt (系统提示指令)。
# 它规定了助手必须把思考过程包裹在 <think> 里，最后给出精确的 <answer>。
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    # rollout 意为利用当前策略来进行多步前向生成（即产生交互回放记录）
    model.eval() # 生成阶段开启 eval() 可以停掉 dropout 操作等，保证数据相对可靠

    # 1. format prompt (结合系统的指令和具体问题将对话模版进行格式化)
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    # 调用大模型官方支持的默认 QA Chat API 聊天模板系统构造长字符串 prompt
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    # 因为只有生成，所以返回为 PyTorch (pt) tensor。左侧 padding 可用于后续生成！
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times (将同一个模型原始问题的输入 prompt 重复扩展至 num_rollouts 倍，例如组大小)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions (真正让大模型去采样并产出答案的阶段了！)
    pad_token_id = tokenizer.eos_token_id
    # 定义生成相关的超属性组合配置。我们开启了 temperature, top_p 和真正的自由 sample (而不是普通的 greedy 生成) 这样才有多样性结果
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    # 调用 generate API 去推理。它会并行输出 num_rollouts 个回答！这就是 GRPO 高效的地方，同题不同答！
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    # 跳过刚刚 prompt 的那部分，仅将新生成的话术 decode 解码后化做自然语言文字列表！
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    # 我们还需要构建 action_mask 以指明在哪里是由机器作答的有效动作（不是填充或 prompt 历史指令）
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    # 机器刚开始生成处向后的位置都是机器回答
    action_mask[:, input_ids.shape[1] :] = True
    # 如果生成的答案短而遇到大量 pad 端终止令牌了，剔除掉不再当作生成区。
    action_mask[sequence_ids == pad_token_id] = False
    # 左移1格代表我们预测下一个的时候是从 1 算起的地方作为实际有效区域(RL中 action 通常对齐到后续生成的那个 token)
    action_mask = action_mask[:, 1:]

    # 3. determine rewards (关键步骤。这是我们的 Reward Model 机制，此处使用的是基于硬匹配的标量奖励而不是深度神经网络)
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float) # 建立该组答案的回报数组记录
    for i, completion in enumerate(completions): # 走遍组内的每一条不同的答案进行打分评级
        # search answer tag (搜寻带有 <answer> 标签的内容格式)
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1).strip() if answer_match else None
        reward = 0
        format_ok = answer_match is not None
        if format_ok:
            reward += 0.1  # format reward (就算算错也会给予 0.1 分作为规范格式遵从的弱激励奖赏！)
            if answer == oracle_answer:
                reward += 1.0  # exact match (如果数学运算等能全字匹配黄金答案则得高分 1.0 满分)
            elif oracle_answer in answer:
                reward += 0.5  # partial match (只要长篇大论内存在着黄金答案，退一步获得 0.5 分残值奖励)

        # 记录每个 rollouts 回答得到的最终得分为后续强化做准备。
        returns[i] = reward

    # 同时返回所有的生成完整数据（历史提示+答案 ID）、回报张量、Action掩码蒙板以及明文文字
    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    # 固定住各种内置的全局随机种子保证结果相对稳定具有一定重现可能
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # DeepSeek GRPO 算法特色！放弃训练估值网络 (Critic)，直接在这单个问题生成的组内的好坏差异之中归纳为“相对优势”打分！
    # 如果它的回报大于组内均值，获得的正优势将会促使其梯度加大、下次更大可能出现；反之得到负优势被抑制。
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    # 对整个词表进行了 softmax 的对数运算，获取全词表的 log 概率
    log_prob = F.log_softmax(logits, dim=-1)
    # 利用 gather 将指定的实际生成 token 的对数概率提取出来，降维。相当于在模型庞大的输出海洋里捞出具体输出 token 的对数概率
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # 这个函数利用语言模型前向推理来获悉每一步在特定上下文和回答情况下，它对选择各个字词的置信概率。
    
    # 手动建立 position_ids 的顺序并屏蔽开所有 pad token （保证计算序列依赖位置关系时跳过填充物）
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    # 开始一整条 sequence_ids 传给前向。这里用 use_cache=False 取消生成推理时避免重算的那一步 KV 缓存机制，防止梯度反向失效
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    # logits 是整个输出矩阵词表分布！
    logits = output["logits"]
    
    # 第一层拿出直到生成部分的最后；使用上置定义好的概率选择工具拿到真实 sequence 的预测生成动作置信度
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    # 便利的生成读取大量按行 JSONL 数据格式工具。防内存炸了
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    # 通过对条件 predicate 设限来加载所需要的数学任务 json 到列表中。比如我们可以规定问题长度不要太长。
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def find_latest_checkpoint(checkpoint_path: Path) -> tuple[Path | None, int]:
    """Find the latest saved checkpoint and return (path, step_number)."""
    checkpoints = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not checkpoints:
        return None, -1
    latest = checkpoints[-1]
    step = int(latest.name.split("_")[1])
    return latest, step


def main():
    seed = 42
    wandb_project = None  # 如果想使用 wandb 图标系统查看实验细节，设置项目名为 "tiny_grpo" 即可。默认关闭。
    device_index = 0      # 获取要指定使用到的 GPU ID，在这里默认为本地的第一块显卡 cuda:0
    resume = True         # ← 设置为 True 则自动从最新存档恢复训练，False 则从头开始

    # ====== Model Configuration (模型整体配置部分) ======
    # 采用的是较小的模型 Qwen2.5-1.5B 才能装载下两份（一份策略、一份基准）模型
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 10                  # 控制要经历多少 RL 大步骤生成一个完整权重存档
    train_batch_size = 8                      # 为迎合本地显存控制。该数值减小有助于防止内存 OOM 溢出爆裂
    lr = 5e-6                                 # RL 微调通常要求 Learning Rate (学习率) 要显著较小。此处 5x10^-6
    kl_weight = 0.01                          # 正如之前 loss 的惩罚项 KL 的比例控制参数
    clip_eps = 0.2                            # PPO 中很经典的截断参数限制阀门

    # ====== GRPO Parameters (GRPO 的特有配置参数) ======
    group_size = 4  # 速度优化：从8减半到4，节省约50%时间，代价是优势估计稍微粗糙
    rollouts_per_step = 8  # 速度优化：从16减半到8，节省约50%时间，代价是训练信号方差稍大
    epochs_per_step = 1     # 对这个步骤获取的新缓存复用的回合。因为是 On Policy 默认给 1，防止 Overfitting 数据偏出。
    max_norm = 1.0          # gradient clipping 参数控制极端的步长

    # ====== Rollout Params (机器自由抽卡/推理探索控制) ======
    max_length = 512        # 控制机器废话时长不要超越。避免无穷延伸的思绪干掉所有的显存容量
    top_p = 1.0             
    temperature = 1.0       

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)          # 将种子固定方便实验定参重跑结果一致性

    # ====== Resume from Checkpoint ======
    resume_checkpoint, resume_step = find_latest_checkpoint(checkpoint_path)
    if resume and resume_checkpoint is not None:
        print(f"[Resume] Found checkpoint at {resume_checkpoint} (step {resume_step}). Resuming from there.")
        policy_path = str(resume_checkpoint)
    else:
        print(f"[Start] Training from scratch (no checkpoint found or resume=False).")
        policy_path = model_name

    print(f"Loading reference model: {model_name}")
    # 构建评判用的参考老师模型（永远用原始预训练权重，不变！）
    reference_model, _ = load_model(model_name, device_map=device)
    print(f"Loading policy model from: {policy_path}")
    # 策略模型：如果 resume，则从存档加载；否则从原始权重加载
    model, tokenizer = load_model(policy_path, device_map=device)
    # Adam 一般被认定是深度学习微调的标准优化器，对参数收敛适应好。
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()  # 让标杆模型乖巧进入不活动的评测层级防阻反向计算图构建（非常关键）！
    
    # 开启“梯度检查点”，用计算时间换取了巨大的内存空间！防止大语言模型几十层导致显存崩溃！这是能在玩家电脑上微调的核心科技
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 从我们的 task JSONL 文件库里面挑选出题目。这定义了 "条件" 函数（Lambda）：尽量挑字数少、计算简单的题目来加速训练看到收敛
    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 2,
        max_rows=8192,
    )
    print(f"found {len(prompts)} matching prompts")
    # 初始化原生标准的 PyTorch 批处理工具 DataLoader 用于将上面获得的一条条 prompt 数组组合分批供给运算图。
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,  # 每个 Epoch 题目顺序会被完全打乱，避免模型死记硬背特定周期的顺序
        drop_last=True,
        pin_memory=False,
    )

    # 实体化一个 Replay 缓冲区
    replay_buffer = ReplayBuffer()
    # 实体化那个包含 GRPO 特定限制器核心业务的 Loss 对象模块。
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled") # 用户若没设定就禁用 wandb 防止报错
    else:
        wandb.init(project=wandb_project)  # 如果有需求则上传

    from tqdm import tqdm
    # 【核心主流程循环】=========================================
    for k, prompt_batch in enumerate(tqdm(prompt_loader, desc="Training Steps")):
        rollout_returns = []

        replay_buffer.clear() # 由于是基于当前策略数据更新算法（On-Policy RL）所以用完一波全抛干净开启新数据回合。

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        # ==== 阶段1：不再建立梯度图去自由飞翔采样回答探索阶段 ====
        with torch.no_grad():
            from itertools import zip_longest 
            for q, a in zip(questions, answers):
                # 利用我们在文件中顶部做的工具，使得系统生成 group_size 个针对同一问题的各自探索的答案分支路线！
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                # 此处就是 GRPO 抛弃了庞大 Critic Model 的神奇核心魔法位置了。根据同组这些返回值做 Z-score 直接标准化拿到相对优势！
                advantages = group_advantages(returns)
                # 定义不等于 pad 令牌位置的地方都纳入有效的大蒙版。
                attention_mask = sequence_ids != pad_token_id

                # 在此拿到了这一步模型本身的原本预测置信度向量字典
                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                # 再用恒定不改变的参考大标杆跑一边，拿到它们心底的旧版本置信度。未来这两个会算惩罚避免跑偏太多。
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                # 记录散度数据参数组（KL Divergence）
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                # 将所有的上述探索结果！完完整整存入这个 Experience 包内并放入缓冲池（将原本的 tensor 转去内存 cpu 免显存爆缸）
                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        # 【阶段1完成】此时探索收集完毕！立即执行系统显存清理（释放 Pytorch 缓存）提供下一步庞大的梯度计算显存余量！
        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        avg_return = torch.stack(rollout_returns).mean()
        print(f"\n{'='*60}")
        print(f"Step {k}: total_returns={episode_return_sum:.4f}, avg_return={avg_return:.4f}")
        print(f"{'='*60}\n")
        wandb.log({"returns": episode_return_sum, "avg_return": avg_return})

        # 构建给第二步提供数据的回溯 DataLoader 处理结构。把刚存下来的纯张量经历装上车跑向训练目标！
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True, # 随机重排打散来破坏数据的相邻依赖性
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        
        # ==== 阶段2：PPO Epochs 基于经验的正式迭代和策略模型改进 ====
        for step_epoch in range(epochs_per_step):
            model.train() # 让所有的 LayerNorm 或是 Dropout 进入了认真训练反向修正梯度图状态。

            for exp in experience_sampler:
                exp: Experience
                
                # 第一时间将显存外的该条条经历搬运回 GPU 的设备上。为快速矩阵并合做好一切准备。
                exp = exp.to(device)

                # 将被系统积累过的上一轮梯度彻底归零（必须项，防止梯度不断叠加产生发散污染）
                optimizer.zero_grad()

                # 通过前向计算取得当前已被学习微调的策略在这批同样的记录经历上最新的分布（在更新后，它在发生偏移）。
                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                # 【执行制裁！】使用 Loss 模块得出它该收到多少改变的损失反馈！包含限幅和 KL！
                loss, kl = objective(log_probs=log_probs, experience=exp)

                # 一旦在混合精度运算中产生不可算值如 NaN 或者 inf 就当场中止，因为它会通过导数毒死全部权重（NaN灾难）。
                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                # Pytorch 神圣的反向传播传播！所有的参数获得了属于本批经验积累下的偏移量规划。
                loss.backward()
                # 并在此进行严密地裁剪控制，任何梯度的 L2 标准范总和超越 max_norm (1.0) 会成比例锁死限制。防止模型发疯剧变
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                # 一切检验通过后正式发车。在优化器步进后将网络所有的神经元微小的权重向正确的方向做出实体位移推演！
                optimizer.step()

        # 等所有的更新和经历结束后，如果跨过了设定跨度即做物理硬盘模型快照保存，确保哪怕断电也不丢失进展
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    # 全部题库刷完，大完结。存下最后的劳动结果模型，宣告收敛闭幕。
    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
