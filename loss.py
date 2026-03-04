from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    蒙特卡洛近似计算 KL 散度，使用 k3 估计量可以更稳健无偏地估算 KL 惩罚，防止策略模型偏离基准模型太远。
    """

    # 计算对数概率的差值，也就是 log(P_ref) - log(P) = log(P_ref / P)
    log_ratio = log_probs_ref.float() - log_probs.float()
    
    if action_mask is not None:
        # 如果存在 action_mask，只保留模型生成的 token 的差异，忽略 prompt 部分
        log_ratio = log_ratio * action_mask

    # k3 估计量公式: exp(log_ratio) - log_ratio - 1 -> 即 (P_ref/P) - log(P_ref/P) - 1
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    # 辅助函数：由于序列中包含了 prompt，或者被 Padding 填充过长度，我们需要使用 mask 对有效的输出进行均值计算
    if mask is None:
        # 如果没有 mask，直接求整个张量的平均值
        return tensor.mean(axis=dim)
    
    # 对被 mask 住的有效部分求和(tensor * mask)，再除以有效元素的个数（即 mask 按该维度的求和），得到 masked 均值
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss (GRPO 策略损失函数)"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        # clip_eps 用于 PPO 截断的超参数，限制策略不要每次偏离更新得太多
        self.clip_eps = clip_eps
        # kl_weight 控制 KL 散度惩罚项权重的超参数
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 获取之前用来生成当前数据的策略的 action 概率分布对数值（即 old_policy）
        old_log_probs = experience.action_log_probs
        # 获取最原始模型的 action 概率分布对数值，作为基准
        log_probs_ref = experience.log_probs_ref
        # 获取有效 action（非 prompt 也非 padding）的掩码
        action_mask = experience.action_mask
        # 从同一个问题生成的多个答案计算出的优势值。正表示该答案在组内较好，负表示较差。
        advantages = experience.advantages

        # 计算当前模型与参考模型之间的 KL 散度，作为惩罚项来控制不可挽回的过度偏移
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        # 重要性采样的比率 (Importance Sampling Ratio)：当前策略产生该动作的概率是原来策略发生此动作的倍数
        # 即 P_new / P_old，等价于 exp(log_probs - old_log_probs)
        ratio = (log_probs - old_log_probs).exp()
        
        # surrogate 1：优势函数未经截断的原始代理目标 (即策略上升的纯粹期望梯度)
        surr1 = ratio * advantages
        
        # surrogate 2：为了防止 ratio 变化过大，导致梯度步长过激被带偏，我们将 ratio 裁切在 [1-eps, 1+eps] 范围内
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        # RL的优化目标是最大化回报，但在 PyTorch 等框架中我们使用最小化 Loss，所以选 min 部分前面加了负号。
        # 加上对改变过度幅度的 KL 惩罚，合成分为 PPO-clip Loss 与 KL-Penalty
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        # 在序列维度上（dim=-1），去除 padding 和 prompt 对应的部分计算这批数据中单个位置上的平均 loss
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        
        # 顺便返回整个批次的 KL 的均值用于外部过程监控（如 wandb 日志，检查是否跑飞）
        return loss, kl.mean()
