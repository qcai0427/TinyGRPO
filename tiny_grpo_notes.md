# 深入理解 Tiny-GRPO 实现

这份笔记详细解析了 `tiny-grpo` 项目的每一步实现机制。该项目展示了如何用极简代码在本地单卡上使用 GRPO（Group Relative Policy Optimization）算法微调大语言模型。

## 核心概念：GRPO 算法概览
GRPO（组相对策略优化）是 DeepSeek 团队提出的一种强化学习算法。传统的 PPO（Proximal Policy Optimization）需要一个与策略模型相同大小的 Critic 模型（Value Model）来估算优势函数。而 GRPO 免去了 Critic 模型：对于同一个问题，它让策略模型生成 $G$ 个不同的回答（组成一个 Group），然后通过对这 $G$ 个回答的奖励（Reward）进行标准化，直接计算相对优势（Advantage）。这极大节省了显存开销，使得在单卡上训练推理模型成为可能。

## 代码结构解析

项目分为三个核心文件：
1. `loss.py` - 定义了 GRPO 的核心损失函数计算逻辑（包含 KL 散度约束和 PPO-clip 损失）。
2. `replay_buffer.py` - 定义了经验回放缓冲区（Replay Buffer），用于存储生成阶段的数据以便在训练阶段进行迭代更新。
3. `train.py` - 核心训练脚本，将模型加载、数据生成（Rollout）、奖励计算、经验存储和模型优化串联起来。

接下来，我们对每一部分进行详细拆解。

### 1. `loss.py`：PPO/GRPO 损失函数

这个文件实现了 actor 的训练目标。与无监督微调（SFT）不同，强化学习通过最大化预期奖励来学习。

#### KL 散度（approx_kl_divergence）
为了防止策略模型（Policy Model）在追求高奖励时过度偏离初始模型（Reference Model），我们需要引入 KL 散度惩罚。
- 公式采用的是 k3 估计量（即 `exp(log_ratio) - log_ratio - 1`）。
- 如果不加以约束，模型可能会很快找到一个只满足规则但不像自然语言的 "捷径" 答案（Reward Hacking）。

#### GRPOLoss
- 计算重要性采样的比率（Importance Sampling Ratio）：`ratio = exp(log_probs - old_log_probs)`。这表达了当前更新后的策略与生成数据时的策略的区别。
- 使用 PPO 经典的 Clip 技巧：确保 Policy 不会一次更新过大。
  `surr1 = ratio * advantages`
  `surr2 = clip(ratio, 1-eps, 1+eps) * advantages`
  `loss = -min(surr1, surr2) + kl_weight * kl`
- 最后通过 Action Mask 对损失取平均（我们只对模型生成的文字这部分求损失，不惩罚 Prompt），完成目标函数计算。

### 2. `replay_buffer.py`：经验数据的组装与管理

在强化学习中，我们要把交互产生的数据（状态、动作、奖励等）存起来用于后续反向传播。
- `Experience` 类：利用 `@dataclass` 声明了一个经验单元，包含了强化学习所需的全部中间张量资源（`log_probs`、`advantages` 等）。
- 每次 Rollout 生成的批次数据会被 `split_experience_batch` 拆分成单个样本存储到 `ReplayBuffer` 中。
- 在训练阶段，又会通过 DataLoader 从 Buffer 采样，并使用 `join_experience_batch` 将不同长度的序列拼接并进行左侧零填充（Left Zero Padding），从而聚合成一个整整齐齐的 Batch 大张量输入给模型算出新的 log_probs。

### 3. `train.py`：核心训练流程

整个算法周期划分为以下几个清晰的步骤：

#### 第一步：加载模型和数据
- 加载因果语言模型（CausalLM）。这里特别加载了两个实例：`reference_model`（评判基准，直接冻结不更新梯度）和 `model`（策略模型，需要被优化）。
- 加载数学问题数据集 `math_tasks.jsonl`，并过滤掉较难的题目（限制 token 和数字长度以加速代码验证）。

#### 第二步：Rollout（探索与生成阶段）
这是 GRPO 的核心特征——为一个 Prompt 采样同组下多个不同的答案：
- **复制输入**：将用户的 Prompt 复制 $G$ 次（对应代码中的 `num_rollouts=group_size`）。
- **批量自由生成**：让模型基于这 $G$ 个一样的 Prompt 开启 `do_sample=True`（带有一定的 temperature）生成 $G$ 种天马行空的回复。
- **提取与定标打分 (Reward)**：利用正则表达式找出 `<answer> ... </answer>` 标签里的内容。通过编写纯 Python 脚本给格式和答案一致性授予不同档次的浮点数积分（例如全答对 1.0，完全没答案 0.0）。这里不需要复杂的深度学习偏好模型奖励评定。

#### 第三步：优势标准化 (Advantage Normalization)
这正是 GRPO 脱离 Critic 模型的秘诀：
把上面一组 $G$ 个打分计算其均值和标准差！然后做简单的组内归一化计算：`(returns - returns.mean()) / returns.std()`。
- 这意味着：在这 $G$ 个答案中，相对较好的答案经过平均化后会天生得到正的优势（Advantage），较差的答案会得到负的优势。
- 用它乘以策略的改变似然数，就能准确指挥模型增加好回答出现率并打压坏回答出现率。

#### 第四步：收集对数似然评估（Log Probabilities）
- 在更新前，评估出生成这批 token 原本旧版的概率 `log_probs` （用来作为比率的分母）和 `log_probs_ref`（用来做防跑偏基准）。这些和打分一起被封装到 `Experience` 中暂存至缓冲池。清理多余系统缓存准备开火！

#### 第五步：策略模型优化（PPO Epochs 更新步）
把 `replay_buffer` 中的数据取出来重新合并 Batch。执行 `epochs_per_step` 遍：
- 进行模型向前传播，获取策略调整过哪怕只有一点点后的最新对数概率 `log_probs_new`。
- 将新旧对数概率比对，运用 PPO 限幅裁剪，加上先前的 `advantages` 参数传给 `GRPOLoss`，计算目标损失导数。
- 启动 `loss.backward()` 计算梯度倒车！期间附上梯度裁剪（Gradient Clipping）抵御梯度失真爆炸。
- 调用 `optimizer.step()` 更新神经网络参数本身。

#### 第六步：日志与保存
- 这个循环重复多次。周期性使用 Weights & Biases（W&B）记录整个过程的 `KL` 和 `Grad Norm`。到预先设定跨度用 `save_pretrained` 保存权重微调。

---
以上就是 `tiny-grpo` 的核心理念与实现脉络。相比大规模复杂的强化学习并行系统，代码极其清爽紧凑，是理解主流逻辑大模型强化强化学习思想的极佳蓝本。项目中的一切文件我已经用中文对每一行做了详尽说明！
