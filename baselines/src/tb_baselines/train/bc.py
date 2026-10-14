"""行为克隆（Behaviour Cloning）：将指定规格等级的网络拟合到教师策略的动作上。

    python -m tb_baselines.train.bc --class micro --data data/teacher.jsonl.gz --epochs 6

这就是**规格天梯（class ladder）**：将单一固定教师蒸馏到五个字节预算等级中，
从而让竞赛真正提出的问题 —— 在 N 字节内能容纳多少博弈智能 —— 得到可衡量的答案，而非主观争论。
它也是整个矩阵中可靠的一半。在 Ants 上进行自对弈强化学习（Self-play RL）面临奖励稀疏、长时序范围、
多智能体协同以及战争迷雾等挑战；而模仿学习则不受这些问题困扰，且天梯在需要有趣的对手之前，首先需要能跑起来的对手。

## 损失函数是针对单只蚂蚁（per ant）计算，而非整个棋盘

网络输出一个稠密的 `[batch, 5, rows, cols]` 策略图，但只有当前座位蚂蚁所在的单元格具有实际意义。
因此，损失函数仅采集这些有蚁单元格，并与教师在对应位置的动作计算交叉熵。
其余所有位置均不受约束，这是刻意为之的设计：一个棋盘有 16,384 个单元格，其中大约只有 30 个有蚂蚁，
如果让空格子也向任何目标训练，会将真实信号淹没在 500:1 的噪声之中。

在阅读损失曲线前值得注意的一个推论：**一个批次（batch）包含的决策样本数量是变化的**，
因为不同对局和座位上的存活蚂蚁数量不同。报告的损失是按每只蚂蚁平均计算的。

## 实测标签分布特征

在 116 万次决策样本中，教师的动作分布为：**西(W) 25.6% / 东(E) 25.3% / 南(S) 24.3% / 北(N) 23.8% / 原地待命(-) 1.0%**。
无需做类别不平衡矫正：蚂蚁几乎总有目的地要奔赴，且四个方向的分布接近均匀。

这种接近均匀的分布，也是第一个数据集被废弃的原因。当时教师在决策前会对方向进行随机打乱，
导致在开阔地形这一常见场景中，两个等距邻居成了掷硬币二选一，四分之一的标签根本不是状态的确定性函数。
对抛硬币计算交叉熵存在任何模型容量都无法突破的下限，因此规格天梯测量的将是教师引入的随机噪声，
而非各规格自身的表达容量。如今教师已是确定性的；详见 `teacher.py`。

准确率是被最小化的优化量，但不是衡量对局实力的最终指标。网络可能在五分之四的情况下与教师一致，
但对战表现却差得多，因为剩下的五分之一错误会在 300 回合中呈复利式累积放大。请务必实战检验。

## `--memory` 参数

启用 `--memory` 时，网络结构为 `nets.WithMemory`：它在读取棋盘特征的同时读取座位的记忆面（`planes.MEMORY`，
通过 `Max` 保留的两个特征面），并将更新后的记忆写回。记忆是当前行之前各回合的确定性函数，
因此训练循环保持不变：`collect.py` 在每行数据的 `m` 键下记录座位当时记住的内容，`tensors` 通过 `planes.encode_memory`
将其渲染为张量。由于教师不读取记忆，标签并不依赖于记忆，记忆模型蒸馏的依然是同一个寻食策略；
此训练运行验证的是带记忆模型贯穿准入校验、时序承接与 `conform` 一致性测试的整套链路。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import nets
from ..export import budget, classes, command
from ..planes import MEMORY, MOVES, encode, encode_memory

MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


def device() -> torch.device:
    """Apple 芯片上使用 MPS，这也是本仓库的开发环境。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


SIZE_RE = re.compile(rb'"size":\[(\d+),(\d+)\]')


class Rows:
    """将原始文本行作为数据集，按批次解析。

    以原始字节（bytes）而非解析后的字典形式持有，这不是微观优化：250,000 条观测数据
    约为 150 MB 文本行，但若全部作为 Python 对象加载，在 16GB 内存的机器上将占用约 1GB 内存。
    在加载时通过正则表达式提取棋盘尺寸，以便在不解析完整 JSON 的情况下按尺寸分组，
    一个大小为 32 的批次只需执行 32 次 `json.loads` —— 相比后续的卷积计算，开销微不足道。
    """

    def __init__(self, path: Path):
        self.lines: list[bytes] = []
        self.sizes: list[tuple[int, int]] = []
        with gzip.open(path, "rb") as f:
            self.header = json.loads(f.readline())
            try:
                for line in f:
                    m = SIZE_RE.search(line)
                    if not m or not line.endswith(b"\n"):
                        continue
                    self.lines.append(line)
                    self.sizes.append((int(m.group(1)), int(m.group(2))))
            except EOFError:
                # 采集中途停止会导致 gzip 流末尾不完整：已完整写入的部分是合法数据集，
                # 而破损的最后一行直接丢弃。
                pass

    def __len__(self) -> int:
        return len(self.lines)

    def parse(self, i: int) -> dict:
        return json.loads(self.lines[i])


def batches(rows: Rows, indices: list[int], size: int, rng: random.Random, shuffle: bool = True):
    """按棋盘尺寸分组的批次生成器。

    按尺寸分组并非图省事：棋盘有多种尺寸，单个张量无法同时容纳两种尺寸。
    打乱操作在每个尺寸组内部进行，因此每个批次始终保持规整，同时每个 epoch 仍能交错遍历不同尺寸组。
    """
    by_size: dict[tuple[int, int], list[int]] = {}
    for i in indices:
        by_size.setdefault(rows.sizes[i], []).append(i)

    chunks = []
    for group in by_size.values():
        if shuffle:
            rng.shuffle(group)
        chunks += [group[i : i + size] for i in range(0, len(group), size)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def tensors(rows: Rows, batch: list[int], dev: torch.device, memory: bool = False):
    """将一个批次转换为 `(boards, memory, sample, row, col, label)`，在其中所有蚂蚁上展开展平。
    除非显式要求，否则 `memory` 为 None：各行对应座位记录的记忆状态，作为计算图的 `memory_in`。"""
    parsed = [rows.parse(i) for i in batch]
    boards = np.concatenate([encode(r["o"]) for r in parsed], axis=0)
    mem = None
    if memory:
        mem = torch.from_numpy(np.concatenate(
            [encode_memory(r.get("m"), r["o"]["size"]) for r in parsed], axis=0)).to(dev)
    b_idx, rr, cc, yy = [], [], [], []
    for i, r in enumerate(parsed):
        for (row, col), ch in zip(r["o"]["mine"], r["a"]):
            b_idx.append(i)
            rr.append(row)
            cc.append(col)
            yy.append(MOVE_INDEX.get(ch, MOVE_INDEX["-"]))
    return (
        torch.from_numpy(boards).to(dev),
        mem,
        torch.tensor(b_idx, dtype=torch.long, device=dev),
        torch.tensor(rr, dtype=torch.long, device=dev),
        torch.tensor(cc, dtype=torch.long, device=dev),
        torch.tensor(yy, dtype=torch.long, device=dev),
    )


def ant_logits(policy: torch.Tensor, b, r, c) -> torch.Tensor:
    """将 `[batch, moves, rows, cols]` 在有蚂蚁的单元格处下投影为 `[ants, moves]`。

    这与适配器的 `out` 程序通过 `tb.gather` 执行的投影完全相同，这也正是重点所在：
    此处训练的优化目标，正是实际对局中决定移动意图的物理量。
    """
    return policy[b, :, r, c]


def run_epoch(model, rows, chunks, dev, opt=None, memory: bool = False) -> tuple[float, float, int]:
    train = opt is not None
    model.train(train)
    total_loss = correct = seen = 0
    for batch in chunks:
        boards, mem, b, r, c, y = tensors(rows, batch, dev, memory)
        if y.numel() == 0:
            continue
        with torch.set_grad_enabled(train):
            # 带记忆模型返回 (policy, memory)；记忆是确定性函数，不参与梯度反传。
            policy = model(boards, mem)[0] if memory else model(boards)
            logits = ant_logits(policy, b, r, c)
            loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # 存活蚂蚁数在 1 到 180 之间浮动，因此批次的梯度尺度也随之大幅波动。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        n = y.numel()
        total_loss += loss.detach().item() * n
        correct += int((logits.argmax(1) == y).sum())
        seen += n
    return total_loss / max(seen, 1), correct / max(seen, 1), seen


def override(spec: dict, args) -> dict:
    """获取规格条目，并用命令行指定的任何参数进行覆盖。"""
    spec = dict(spec)
    for key in ("arch", "channels", "blocks"):
        value = getattr(args, key, None)
        if value is not None:
            spec[key] = value
    # `percell` 和 `trunk` 不接受 stride 参数；若残留该参数，会在长时间训练结束后的构造函数中报错。
    if spec.get("arch") in ("trunk", "percell"):
        spec.pop("stride", None)
    return spec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    # 变体是命令行参数，而非第二张静态表：运行的 history.json 与模型卡片都会记录实际构建的内容，这才是关键记录。
    ap.add_argument("--arch", default=None, help="覆盖该规格的默认网络架构")
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--memory", action="store_true",
                    help="在棋盘旁额外携带 planes.MEMORY（数据集也必须携带记忆）")
    ap.add_argument("--data", type=Path, default=Path("data/teacher.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = device()

    rows = Rows(a.data)
    header = rows.header
    if a.memory and header.get("memory") != [m.name for m in MEMORY]:
        raise SystemExit(
            f"{a.data} carries {header.get('memory') or 'no'} remembered state and --memory needs "
            f"{[m.name for m in MEMORY]}: collect it again (`python -m tb_baselines.collect`)"
        )
    # 并且教师必须已读取过记忆。无论是否启用记忆，`collect` 都会在每个数据集上写入 `memory` 平面名称，
    # 因此仅凭该键无法证明任何事 —— 它仅表明数据行携带这些列。
    # `teacher_remembers` 才是关键凭证，专门为此检查而写入：若无此标志，标签来自无记忆教师，
    # `--memory` 将训练模型去参考一个其目标策略从未参考的状态，而 `train.py` 却会打上
    # "在带记忆教师标签上进行行为克隆" 的标签。模型将学会忽略记忆，且没有任何报错提示。
    if a.memory and not header.get("teacher_remembers"):
        raise SystemExit(
            f"{a.data} was collected from a teacher that does not remember, so --memory would train "
            f"on labels that never read the memory: collect it again with "
            f"`python -m tb_baselines.collect --remember`"
        )
    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = int(len(order) * (1 - a.holdout))
    train_idx, val_idx = order[:cut], order[cut:]

    spec = override(classes()[a.cls], a)
    model = nets.build(spec, memory=a.memory).to(dev)
    b = budget(a.cls)
    print(f"{a.cls}: {nets.policy_params(model):,} parameters "
        f"(budget about {b['params_at_target']:,}), {dev.type}")
    print(f"data: {len(train_idx):,} train / {len(val_idx):,} held out, "
        f"engine {header['engine_digest'][:20]}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    val_chunks = batches(rows, val_idx, a.batch, rng, shuffle=False)

    out = a.out or Path("runs") / (f"{a.cls}-bc-memory" if a.memory else f"{a.cls}-bc")
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        chunks = batches(rows, train_idx, a.batch, rng)
        tl, ta, n = run_epoch(model, rows, chunks, dev, opt, a.memory)
        vl, va, _ = run_epoch(model, rows, val_chunks, dev, memory=a.memory)
        sched.step()
        took = time.time() - t0
        print(f"  epoch {epoch}/{a.epochs}  train {tl:.4f} / {ta:.1%}   "
            f"held out {vl:.4f} / {va:.1%}   {n:,} ants  {took:.0f}s")
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va, "seconds": round(took, 1)})
        if vl < best:
            best = vl
            torch.save({"trunk": model.state_dict(), "class": a.cls, "memory": a.memory,
                        "engine_digest": header["engine_digest"]}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(
        {"class": a.cls, "method": "bc", "arch": spec["arch"], "spec": spec, "memory": a.memory,
        "data": str(a.data), "seed": a.seed, "command": command("tb_baselines.train.bc", argv),
        "epochs": history,
        "engine_digest": header["engine_digest"], "device": dev.type}, indent=2) + "\n")
    print(f"  -> {out / 'best.pt'}  (best held-out loss {best:.4f})")
    print("  accuracy against the teacher is not strength. Play it: `python -m tb_baselines.eval`")


if __name__ == "__main__":
    main()
