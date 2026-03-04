from dataclasses import dataclass, fields
from typing import Optional
from typing_extensions import Self

import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    # 辅助函数：用来在批处理过程中将不同长度的张量序列补齐到相同的长度
    assert side in ("left", "right") # 只允许在左侧或右侧进行 padding
    # 找到这个序列列表里最大的序列长度
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        # 计算当前序列需要补足多少个0才能达到最大长度
        pad_len = max_len - seq.size(0)
        # 如果是左侧 padding（适合自回归生成模型例如大语言模型），在前面补 pad_len 个0，后面补 0 个0
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        # 用 F.pad 方法在序列一维填充
        padded_sequences.append(F.pad(seq, padding))
    # 最后将它们沿着 batch 维度拼接成一个整体张量 Tensor
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    # 表示一段训练所需 "经验" 数据结构的容器（包括生成的 token 以及用于 RL 损失计算关联的值）
    sequences: torch.Tensor          # 回答的真实 Token ID（包含 prompt 或是由 tokenizer 处理过的格式）
    action_log_probs: torch.Tensor   # 模型采样生成 token 时的动作概率的对数值（Old policy）
    log_probs_ref: torch.Tensor      # Base 冻结模型生成该 token 所对应的概率分布，用于 KL 限制
    returns: Optional[torch.Tensor]  # 这条路径或这段生成的绝对奖励分数
    advantages: Optional[torch.Tensor] # 去均值归一化后的相对优势值，正的表示比组内平均好，负表示不如平均好
    attention_mask: Optional[torch.Tensor] # 判断是否是 padding token（0 表示 padding 不用给 attention)
    action_mask: torch.Tensor        # 判断当前位置是否属于模型生成（过滤掉 prompt，即只在机器回复区域算 RL 损失）
    kl: Optional[torch.Tensor] = None # 事后计算和存储生成的 KL 散度信息

    def to(self, device: torch.device) -> Self:
        # 将一个 Experience 中的所有 Tensor 数据搬运到指定的硬件设备（如 cuda 或者 cpu）
        members = {}
        for field in fields(self):
            # 获取这个经验中这个属性对应的值
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                # 如果是 PyTorch Tensor 则搬运到 device 上
                v = v.to(device=device)
            members[field.name] = v
        # 创建并返回一个新的存放在对应设备的对象
        return Experience(**members)


def split_experience_batch(experience: Experience) -> list[Experience]:
    # 用于在保存和拆解经验缓冲池时，把合并的张量数据按照 batch 维度（通常为第一维）拆分成单条。
    batch_size = experience.sequences.size(0)
    # 准备 batch_size 个空字典用于存放单个样本
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key) # 获取批处理张量的整体属性
        if value is None:
            # 如果某个属性原本就是 None，那么每条解出来的也理应当是 None
            vals = [None] * batch_size
        else:
            # torch.unbind 沿着第一维切割张量，返回张量构成的元组列表
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            # 填入各个对应的样本结构字典中
            batch_data[i][key] = v

    # 重组后返回成一个个独立的 Experience 列表
    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    # 与前一步反过来的操作。用于在使用 DataLoader 时获取到的若干条分离 Experience 打包成 Batch 大张量
    batch_data = {}
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        # 提取出属于某一特定属性的所有样本集合列表
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            # 为左侧进行补齐 0，使其变成长度统一、整整齐齐并且可以堆叠计算的 Batch 张量
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        # 保存到大字典中
        batch_data[key] = data
    # 实例化一个新的经过组合后的大号 Experience
    return Experience(**batch_data)


class ReplayBuffer:
    # 一个经验缓冲池，暂时缓存生成的轨迹或体验数据（Experience），后续用于反采样进行 PPO/RL 的梯度更新计算
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit               # 池子大小上限。如果传 > 0，则为先进先出结构存储
        self.items: list[Experience] = [] # 真正存放 Experience 节点的列表

    def append(self, experience: Experience) -> None:
        # PPO 或者算法前向步骤得到新的整批经历后，将其打散成单个实例进行附加添加
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            # 若超出大小限制，仅保留列表中靠后（最新）的数据项
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        # 每一次全量 step 更新跑完后清空缓存（因为 PPO 等一般为 On-Policy 策略，只吃当前分布产出的经历）
        self.items.clear()

    def __len__(self) -> int:
        # 方便 DataLoader 或者取长度工具直接获取缓冲池拥有包含条数
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        # 方便用下标访问单个经历，这样 DataLoader 才可以使用。
        return self.items[idx]
