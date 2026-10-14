"""PPO 自对弈强化学习：通过奖励信号而非模仿教师来学习 Ants 策略。

    python -m tb_baselines.train.ppo --class micro --iters 200

这是矩阵中极具趣味性但同时也伴随风险的一半。Ants 具有奖励稀疏、长时序范围、
多智能体协同以及战争迷雾等特点；行为克隆完全不受这些困扰。在盲信不断上升的回报值之前，请先查看 `eval.py` 的对战结果表。

## 两个座位采用相同策略，这正是核心所在

每场对局中，学习者均与自己对战，因此一场比赛产出两条轨迹，且对手的进化速度与学习者完全同步。
这也意味着观测必须是**对称的** —— 如今已确实如此：引擎提交 `sha256:f17b51b6c92b`
将视野中的所有者标签调整为相对于观察者，而在此之前，同一场对局的两个座位实际上在悄然经历两场不同的游戏。

## 动作空间是因子化的，而非联合分布

单次前向传播中，一个座位最多指挥 180 只蚂蚁。联合动作空间高达 5^180，
但网络输出的并非联合分布：它输出的是单元格级别的分布，每只蚂蚁读取自身所在的单元格。因此：

    log pi(a | s) = 所有存活蚂蚁 sum(log pi(a_i | s, pos_i))

这是标准的因子化策略（factorised policy），策略熵也以相同方式累加求和。
需要铭记的一个推论是：**座位的对数概率与其蚂蚁数量呈正比缩放**，
因此拥有 90 只蚂蚁的蚁群所产生的梯度信号是 3 只蚂蚁蚁群的 30 倍。
PPO 目标函数中的重要性采样比率（ratio）正是因此按“单蚁（per ANT）”而非按“单座（per seat）”构建：
按座位计算的比率是 `exp(90 项对数概率差之和)`，这在第一次参数更新时就会冲破截断区间（clip range），
导致算法形同虚设。

## Critic 拥有特权信息且在最终阶段被舍弃

`tinybrains env --scores every` 每个回合都会报告各场对局的真实得分 —— `f_finish` 为进行中的比赛计算该值 ——
且价值头（value head）会读取该得分。这是非对称 Actor-Critic（asymmetric actor-critic），
而非迷雾作弊：`export.py` 最终仅导出主干网络和策略头，任何接触过得分的模块都不会残留到实际参赛构件中。

## 采集轨迹仅保存原始观测，而非完整棋盘张量

一个 `[7, 128, 128]` 的 int8 棋盘占用 114 KiB，在 100 个活跃座位上进行 64 步的采集将产生 730 MB 的张量。
而背后的原始观测字典每条仅约 2 KB。我们在参数更新阶段重新编码张量，每个 epoch 仅耗时约 1 秒，
却让轨迹长度成为自由可调的超参数，而非受制于内存预算。

## 吞吐量取决于张量形状与等待开销，而非算术浮点计算

在开发所用的 M2 Pro 上，micro 在 MPS 上前向+后向传播一块 128x128 棋盘耗时 2.7 ms（fp32；CPU 则慢 10 倍），
这意味着单次迭代的更新理论上仅需约 10 秒的算术计算 —— 但实际耗时却长达 24 秒。
差距源于两点，且在性能分析（profile）中都不会显露为自身，而是体现为下一次访问设备的操作耗时：

- **MPS 为每个张量形状单独编译计算图。** 单蚁张量在几乎每次调用时都有全新的长度，
  因为小批次（minibatch）中的蚂蚁总数几乎从不重复：面对全新长度耗时 153 ms，面对已知长度仅 60 ms。
  现在它们被统一填充（pad）到 `_bucket` 整数倍，且填充权重为 0。
- **向 MPS 的阻塞式拷贝会等待此前队列中的所有操作完成。** 每个小批次以往都要搬运 8 个数组并用 `float()`
  读回 4 项统计指标，导致每次搬运都在等待上一个小批次的后向传播。
  现在 `_Group` 在每次更新时只将同一棋盘尺寸的数据搬移一次，循环体内部仅对已驻留张量做切片。

消除这两项瓶颈后，更新时间回归到了纯算术开销（约 10 秒），48 回合迭代的耗时从最初的
23 / 34 / 39 秒（随着蚁群壮大、未知长度增多而膨胀）降至 13 / 13 / 14 秒。
剩下的全部是卷积运算。`--amp`（fp16 自动混合精度）实测可再提速 8%，但默认关闭：
其梯度与 fp32 相比有约 1% 的微小漂移，对于有限的加速而言并非微不足道。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import nets
from ..env import Env, orders_from_indices
from ..export import budget, classes
from ..planes import N_MOVES
from .bc import device

ANT_BUCKET = 256


def _bucket(n: int, q: int = ANT_BUCKET) -> int:
    """将 `n` 向上取整为 `q` 的倍数，且绝不为零。

    MPS 为每个张量形状缓存编译后的计算图，因此长度不断变化的单蚁张量会导致每次调用都重新编译。
    在一批 32 块 96x96 棋盘的小批次上实测：遇到已见过的长度耗时 59.6 ms，全新长度耗时 153.5 ms，
    填充至 256 的倍数时耗时 57.6 ms（填充到 128 仍不够充分：耗时 69.4 ms）。
    填充行在所有求和运算中均被掩码剔除，因此数学等价性完全一致。
    """
    return max(q, -(-n // q) * q)


def _autocast(dev: torch.device, enabled: bool):
    """仅为主干网络启用 fp16 激活值，其余部分一概不启用：
    每个调用者在经过 softmax、比率计算或损失函数前，均将 logits 和 value 转回 fp32。"""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(dev.type, dtype=torch.float16)


@dataclass
class Reward:
    """单个回合的奖励加权配置。所有项均为训练阶段的主观设定，而非天梯规则本身。

    天梯最终结算的是排名，排名源于得分：摧毁敌方蚁穴 +2，损失自身蚁穴 -1。
    这一信号在 300 回合中仅零星出现几次，因此设计这些塑形奖励（shaping reward）
    是为了让前 1000 次参数更新拥有明确的学习驱动。将它们集中声明在此处而非埋在代码深处，
    以便运行卡片能够显式记录它们。
    """

    score: float = 1.0          # 自身逐回合得分差
    rival: float = 0.5          # 减去对手得分差：Ants 并非纯零和博弈，但攻占蚁穴具有零和性质
    ants: float = 0.02          # 食物经济收益，这是全局唯一的稠密奖励信号
    win: float = 1.0            # 对局终局奖励，基于名次而非分差
    gamma: float = 0.997        # 对应约 300 回合的时序视界
    lam: float = 0.95


@dataclass
class Trajectory:
    """单个座位的轨迹，以 `(ep, seat)` 为键。`ep` 在波次填充（wave refill）跨度下保持稳定，而 `(w, m)` 则不然。"""

    obs: list[dict] = field(default_factory=list)
    # 轨迹采集阶段保留已编码棋盘。`encode` 开销并不为零，否则参数更新阶段每个 epoch 都要对每个样本重新编码 ——
    # 这些重复计算在动作选择阶段本已执行过。
    boards: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    last_score: int = 0
    last_rival: int = 0
    last_ants: int = 0
    primed: bool = False        # Ants 开局每个座位得分为 1，因此首个增量不可计入
    finished: bool = False


def gae(rewards: list[float], values: list[float], bootstrap: float, r: Reward):
    """单个座位的广义优势估计（GAE）与回报值。在真正的终止状态下 `bootstrap` 为 0，
    在轨迹采集截断时为 V(s_T) —— 将两者混为一谈会导致截断对局向智能体传递“世界终结”的错误信号。"""
    adv, out = [], 0.0
    nxt = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + r.gamma * nxt - values[t]
        out = delta + r.gamma * r.lam * out
        adv.append(out)
        nxt = values[t]
    adv.reverse()
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


class Runner:
    """推演采集循环：输入环境，产出轨迹。"""

    def __init__(self, model: nets.ActorCritic, env: Env, dev: torch.device, reward: Reward,
                 seed: int = 0, amp: bool = False):
        self.model = model
        self.env = env
        self.dev = dev
        self.reward = reward
        self.amp = amp
        # 动作在宿主端由其专属随机流采样，使 update 的小批次置换保持在全局 numpy 随机流中。
        self.rng = np.random.default_rng(seed)
        self.step = env.reset()
        self.open: dict[tuple[int, int], Trajectory] = {}
        self.done: list[Trajectory] = []
        self.returns: list[float] = []

    @torch.no_grad()
    def act(self, boards: np.ndarray, rows: "_AntIndex"):
        """为同组中每个棋盘上的每只蚂蚁采样一个动作。

        蚂蚁所在单元格的策略为对应位置的 `softmax(logits)` —— 亦即 `update` 进行微分的因子化策略 ——
        并在宿主端通过累积分布函数（CDF）逆变换进行精确采样。
        在 MPS 上调用 `torch.distributions.Categorical` 需要先跑 `multinomial` 等待前向传播，
        再经历两次数据回传；当前实现仅需一次回传。
        """
        t = torch.from_numpy(boards).to(self.dev)
        with _autocast(self.dev, self.amp):
            logits, values = self.model(t)
        per_ant = logits[rows.board, :, rows.r, rows.c].float()      # [_bucket(n), moves]
        logp_all = torch.log_softmax(per_ant, dim=-1)
        host = torch.cat([logp_all.reshape(-1), values.float()]).cpu().numpy()
        cut = logp_all.numel()
        logp = host[:cut].reshape(-1, N_MOVES)[: rows.n]
        cdf = np.cumsum(np.exp(logp.astype(np.float64)), axis=1)
        u = self.rng.random(rows.n) * cdf[:, -1]
        # 累积概率首次超过 u 的动作；绝不选择概率为 0 的动作
        picks = np.minimum((cdf <= u[:, None]).sum(axis=1), N_MOVES - 1)
        return picks, logp[np.arange(rows.n), picks], host[cut:]

    def collect(self, steps: int) -> dict:
        """执行 `steps` 个环境回合。返回统计指标；轨迹在 `self` 上累积。"""
        self.done, self.returns = [], []
        for _ in range(steps):
            actions: list[str | None] = [None] * len(self.step.seats)

            for group in self.step.groups:
                rows = _AntIndex(self.step.seats, group.indices).to(self.dev)
                picks, logp, values = self.act(group.boards, rows)
                orders = orders_from_indices(picks, rows.counts)

                at = 0
                for k, i in enumerate(group.indices):
                    n = rows.counts[k]
                    actions[i] = orders[k]
                    self._record(self.step.seats[i], group.boards[k : k + 1], picks[at : at + n],
                                 logp[at : at + n], float(values[k]))
                    at += n

            prev = self.step
            self.step = self.env.step([a if a is not None else "" for a in actions])
            self._reward(prev)
            self._close(self.step.ended)

        # 仍在进行的对局进行价值自举（bootstrap），而非假装已结束。
        return {
            "episodes": len(self.returns),
            "mean_return": float(np.mean(self.returns)) if self.returns else 0.0,
        }

    def _record(self, seat, board, picks, logp, value) -> None:
        key = (seat.ep, seat.seat)
        traj = self.open.setdefault(key, Trajectory())
        if not traj.obs:
            traj.last_ants = len(seat.obs["mine"])
        traj.obs.append(seat.obs)
        traj.boards.append(board)
        traj.actions.append(picks.copy())
        traj.logp.append(logp.copy())
        traj.values.append(value)

    def _reward(self, prev) -> None:
        """根据环境报告的得分和维持的蚁群规模，为每个座位计算单步奖励。"""
        r = self.reward
        for seat in prev.seats:
            key = (seat.ep, seat.seat)
            traj = self.open.get(key)
            if traj is None or len(traj.rewards) >= len(traj.obs):
                continue
            scores = self.step.scores.get(seat.ep) or {e.ep: e.scores for e in self.step.ended}.get(seat.ep)
            if scores is None:
                traj.rewards.append(0.0)
                continue
            mine = scores[seat.seat]
            rival = max(s for i, s in enumerate(scores) if i != seat.seat)
            ants = len(seat.obs["mine"])
            if not traj.primed:
                # Ants 开局每个座位有 1 分且初始有 1 只蚂蚁。若从 0 开始会奖励其开局入场。
                traj.last_score, traj.last_rival, traj.last_ants = mine, rival, ants
                traj.primed = True
            value = (
                r.score * (mine - traj.last_score)
                - r.rival * (rival - traj.last_rival)
                + r.ants * (ants - traj.last_ants)
            )
            traj.last_score, traj.last_rival, traj.last_ants = mine, rival, ants
            traj.rewards.append(value)

    def _close(self, ended) -> None:
        for e in ended:
            for seat in range(e.seat_count):
                traj = self.open.pop((e.ep, seat), None)
                if traj is None or not traj.obs:
                    continue
                best = min(e.ranks)
                outcome = 1.0 if e.ranks[seat] == best and e.ranks.count(best) == 1 else (
                    0.0 if e.ranks[seat] == best else -1.0
                )
                while len(traj.rewards) < len(traj.obs):
                    traj.rewards.append(0.0)
                traj.rewards[-1] += self.reward.win * outcome
                traj.finished = True
                self.done.append(traj)
                self.returns.append(float(sum(traj.rewards)))

    def harvest(self) -> list[Trajectory]:
        """产出已完成的对局轨迹，并对仍在进行的对局进行快照，使长对局也能贡献样本。
        未完成的轨迹保留其当前累计得分，避免重复采集，且下一个片段的首个增量从本次停止处继续起算。"""
        batch = list(self.done)
        for key, traj in list(self.open.items()):
            if len(traj.rewards) >= len(traj.obs) and traj.obs:
                batch.append(traj)
                self.open[key] = Trajectory(
                    last_score=traj.last_score, last_rival=traj.last_rival,
                    last_ants=traj.last_ants, primed=True,
                )
        return batch


class _AntIndex:
    """将同组中所有蚂蚁展平成扁平的 `(board, row, col)`，
    这正是策略头与适配器的 `out` 程序共同执行的投影操作。

    填充至 `_bucket(n)` 行，使 gather 操作匹配 MPS 此前见过的形状；
    填充行读取棋盘 0 的 (0, 0) 单元格，在数据传回后由 `act` 丢弃。"""

    def __init__(self, seats, indices):
        mine = [np.asarray(seats[i].obs["mine"], dtype=np.int64).reshape(-1, 2) for i in indices]
        self.counts = [len(m) for m in mine]
        self.n = sum(self.counts)
        idx = np.zeros((3, _bucket(self.n)), dtype=np.int64)
        if self.n:
            idx[0, : self.n] = np.repeat(np.arange(len(mine)), self.counts)
            idx[1:, : self.n] = np.concatenate(mine).T
        self._idx = torch.from_numpy(idx)
        self.board, self.r, self.c = self._idx

    def to(self, dev):
        self._idx = self._idx.to(dev)
        self.board, self.r, self.c = self._idx
        return self


@dataclass
class _Minibatch:
    boards: torch.Tensor        # [B, planes, rows, cols] int8
    board: torch.Tensor         # [M] 该蚂蚁属于哪块棋盘
    r: torch.Tensor             # [M]
    c: torch.Tensor             # [M]
    act: torch.Tensor           # [M] 分配给它的动作
    ret: torch.Tensor           # [B]
    old_lp: torch.Tensor        # [M]
    adv: torch.Tensor           # [M] 其座位的优势估计，广播展开
    mask: torch.Tensor          # [M] 真实蚂蚁为 1，填充到 `_bucket` 的部分为 0
    n: int                      # 真实蚂蚁数，不计填充


class _Group:
    """同一棋盘尺寸下的所有样本，每次更新仅向设备传输一次。

    棋盘以及每个单蚁数组只搬运一次；每个 epoch 的小批次索引计划以两个数组形式搬运；
    循环体自身仅切片已驻留在设备上的张量，避免内部产生宿主端等待。
    """

    def __init__(self, samples, advs: np.ndarray, members: list[int], dev):
        self.n = len(members)
        mine = [np.asarray(samples[i][0]["mine"], dtype=np.int64).reshape(-1, 2) for i in members]
        self.counts = np.array([len(m) for m in mine], dtype=np.int64)
        self.starts = np.cumsum(self.counts) - self.counts
        rc = np.concatenate(mine)
        self.r, self.c = rc[:, 0], rc[:, 1]
        self.act = np.concatenate([samples[i][2] for i in members]).astype(np.int64)
        self.old_lp = np.concatenate([samples[i][3] for i in members]).astype(np.float32)
        self.adv = np.repeat(advs[members], self.counts)
        self.ret = np.array([samples[i][5] for i in members], dtype=np.float32)
        self.boards = torch.from_numpy(
            np.concatenate([samples[i][1] for i in members], 0)).to(dev)

    def minibatches(self, order: np.ndarray, size: int, dev):
        """当前 epoch 的小批次，按 `order` 顺序每次提取 `size` 个样本。"""
        ints, flts, cuts = [], [], []
        for start in range(0, len(order), size):
            sel = order[start : start + size]
            cnt = self.counts[sel]
            n = int(cnt.sum())
            m = _bucket(n)
            ant = np.arange(n) + np.repeat(self.starts[sel] - (np.cumsum(cnt) - cnt), cnt)
            ib = np.zeros(len(sel) + 4 * m, dtype=np.int64)
            ib[: len(sel)] = sel
            per = ib[len(sel):].reshape(4, m)
            per[0, :n] = np.repeat(np.arange(len(sel)), cnt)
            per[1, :n] = self.r[ant]
            per[2, :n] = self.c[ant]
            per[3, :n] = self.act[ant]
            fb = np.zeros(len(sel) + 3 * m, dtype=np.float32)
            fb[: len(sel)] = self.ret[sel]
            fper = fb[len(sel):].reshape(3, m)
            fper[0, :n] = self.old_lp[ant]
            fper[1, :n] = self.adv[ant]
            fper[2, :n] = 1.0
            ints.append(ib)
            flts.append(fb)
            cuts.append((len(sel), m, n))

        ints_t = torch.from_numpy(np.concatenate(ints)).to(dev)
        flts_t = torch.from_numpy(np.concatenate(flts)).to(dev)
        io = fo = 0
        for b, m, n in cuts:
            iv = ints_t[io : io + b + 4 * m]
            fv = flts_t[fo : fo + b + 3 * m]
            io += b + 4 * m
            fo += b + 3 * m
            board, r, c, act = iv[b:].view(4, m)
            old_lp, adv, mask = fv[b:].view(3, m)
            yield _Minibatch(self.boards.index_select(0, iv[:b]), board, r, c, act,
                             fv[:b], old_lp, adv, mask, n)


def update(model, opt, batch: list[Trajectory], r: Reward, dev, epochs: int,
           clip: float, entropy: float, minibatch: int, amp: bool = False,
           scaler: "torch.amp.GradScaler | None" = None) -> dict:
    """PPO 算法更新，重要性采样比率按单蚁（per ant）构建。

    按座位构建的比率将是 `exp(90 只蚂蚁的对数概率差之和)`，在第一次更新时便会冲出任何合理的截断区间。
    按单蚁计算既符合因子化策略的数学定义，也是在数值上唯一能正常起效的版本。

    单蚁对数概率与策略熵直接展开为 `log_softmax` 与 `-sum(p log p)`，
    这与 `torch.distributions.Categorical` 的计算逻辑一致；均值通过真实蚂蚁的有效掩码求和除以蚂蚁数量得到。
    统计指标保留在设备上，直到最后一个小批次处理完毕。
    """
    samples = []
    for traj in batch:
        adv, ret = gae(traj.rewards, traj.values, 0.0 if traj.finished else traj.values[-1], r)
        for t in range(len(traj.obs)):
            if len(traj.actions[t]) == 0:
                continue
            samples.append((traj.obs[t], traj.boards[t], traj.actions[t], traj.logp[t],
                            adv[t], ret[t]))
    if not samples:
        return {"samples": 0}

    advs = np.array([s[4] for s in samples], dtype=np.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        by_size[tuple(sample[0]["size"])].append(i)
    groups = [_Group(samples, advs, members, dev) for members in by_size.values()]

    rows = []
    for _ in range(epochs):
        for group in groups:
            for mb in group.minibatches(np.random.permutation(group.n), minibatch, dev):
                with _autocast(dev, amp):
                    logits, values = model(mb.boards)
                per_ant = logits[mb.board, :, mb.r, mb.c].float()
                logp = torch.log_softmax(per_ant, dim=-1)
                new_lp = logp.gather(1, mb.act.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(new_lp - mb.old_lp)
                policy_loss = -(torch.min(
                    ratio * mb.adv, torch.clamp(ratio, 1 - clip, 1 + clip) * mb.adv
                ) * mb.mask).sum() / mb.n
                value_loss = F.mse_loss(values.float(), mb.ret)
                # 像 Categorical 一样进行截断，使 -inf 对数概率乘以 0 概率得到 0 而非 NaN。
                p_log_p = logp.exp() * logp.clamp(min=torch.finfo(logp.dtype).min)
                ent = (-p_log_p.sum(-1) * mb.mask).sum() / mb.n
                loss = policy_loss + 0.5 * value_loss - entropy * ent

                opt.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(opt)
                    scaler.update()

                clipped = (((ratio - 1).abs() > clip).float() * mb.mask).sum() / mb.n
                rows.append(torch.stack([policy_loss, value_loss, ent, clipped]).detach())
    if not rows:
        return {"samples": len(samples)}

    means = torch.stack(rows).cpu().numpy().astype(np.float64).mean(axis=0)
    return dict(zip(("policy", "value", "entropy", "clipped"), map(float, means))) | {
        "samples": len(samples)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=48, help="每次迭代的环境回合数")
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true",
                    help="主干网络前向和后向传播使用 fp16 autocast 与 GradScaler；"
                         "softmax、比率、损失与优化器保持 fp32")
    ap.add_argument("--init", type=Path, default=None, help="用于初始化的 bc 检查点路径")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()

    spec = classes()[a.cls]
    trunk = nets.build(spec)
    if a.init:
        state = torch.load(a.init, map_location="cpu")
        trunk.load_state_dict(state.get("trunk", state))
    model = nets.ActorCritic(trunk).to(dev)

    b = budget(a.cls)
    method = "bc-ppo" if a.init else "ppo"
    print(f"{a.cls} {method}: {nets.policy_params(model):,} policy parameters "
          f"(budget about {b['params_at_target']:,}), {dev.type}{' fp16 autocast' if a.amp else ''}")

    out = a.out or Path("runs") / f"{a.cls}-{method}"
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(dev.type) if a.amp else None
    reward = Reward()

    history = []
    best = -1e9
    with Env(waves=a.waves, matches_per_wave=a.matches_per_wave,
             max_turns=a.max_turns, seed=1000 + a.seed) as env:
        runner = Runner(model, env, dev, reward, seed=a.seed, amp=a.amp)
        print(f"engine {env.engine_digest[:20]}")
        for it in range(1, a.iters + 1):
            t0 = time.time()
            rolled = runner.collect(a.rollout)
            batch = runner.harvest()
            t1 = time.time()
            stats = update(model, opt, batch, reward, dev, a.epochs, a.clip,
                           a.entropy, a.minibatch, amp=a.amp, scaler=scaler)
            took = time.time() - t0
            row = {"iter": it, **rolled, **stats, "seconds": round(took, 1),
                   "collect_seconds": round(t1 - t0, 1)}
            history.append(row)
            print(f"  {it:4d}  return {rolled['mean_return']:+7.2f} over {rolled['episodes']:3d} eps"
                  f"   pi {stats.get('policy', 0):+.4f}  V {stats.get('value', 0):.3f}"
                  f"  H {stats.get('entropy', 0):.3f}  {stats['samples']:5d} samples  {took:.0f}s"
                  f" (collect {t1 - t0:.0f}s)")
            if rolled["episodes"] and rolled["mean_return"] > best:
                best = rolled["mean_return"]
                torch.save({"trunk": trunk.state_dict(), "class": a.cls,
                            "engine_digest": env.engine_digest}, out / "best.pt")
            if it % 10 == 0:
                (out / "history.json").write_text(json.dumps(
                    {"class": a.cls, "method": method, "reward": vars(reward), "amp": a.amp,
                     "iters": history}, indent=2) + "\n")

    (out / "history.json").write_text(json.dumps(
        {"class": a.cls, "method": method, "reward": vars(reward), "amp": a.amp,
         "iters": history}, indent=2) + "\n")
    print(f"  -> {out / 'best.pt'}")
    print("  a rising return is not strength. Play it: `python -m tb_baselines.eval`")


if __name__ == "__main__":
    main()
