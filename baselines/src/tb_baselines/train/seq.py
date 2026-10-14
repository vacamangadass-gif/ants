"""时序序列上的行为克隆：网络自主学习记忆特征。

    python -m tb_baselines.train.seq --class nano --data data/teacher-memory.jsonl.gz --epochs 5

`train/bc.py --memory` 携带的记忆是数据行前序回合的确定性函数（`planes.MEMORY`，通过 `Max` 保留）：
训练器在 numpy 中预先计算好，因此各行保持互相独立，训练循环与 bc.py 完全一致。
而本训练器携带由计算图自主计算的记忆（`nets.LearnedMemory`）：某一行对应的记忆是网络在前一行输出的自身状态，
因此单个座位的各行必须按时序顺序回放，这正是数据集中包含 `k` 键（`collect.py`）的原因。

## 截断式沿时间反向传播（Truncated BPTT）

一个批次包含具有相同棋盘尺寸的若干座位，从每个座位的首行开始同步步进。
每隔 `--bptt` 个回合，对累积的损失进行反向传播，优化器更新参数并将记忆张量截断分离（detach），
因此梯度的回溯视界为 `bptt` 个回合，而记忆本身则可一直溯源至该座位的首个回合。
各座位对局结束的时机不同；超过最后一行的座位会从批次中移出，且批次内聚合的是长度相近的座位，因此很少会过早移出。

**`bptt` 决定了优化器的更新步数，而该步数直接决定了训练效果。**
每 `batch` 个座位的 `bptt` 个回合更新一步，意味着每个 epoch 执行 `250,000 / (batch x bptt)` 步，
相比之下 bc.py 为 `250,000 / 32` 步：在 16 个座位、32 个回合下更新步数仅为其九分之一，
导致首个 epoch 结束时一致率仅为 45.6%，而 bc.py 首轮即接近 78%。
若设为 8 个回合，并将学习率翻倍以匹配更大的等效批次，首轮结束一致率达到 68%，此后持续爬升。
8 个回合对于这种记忆机制已经足够，因为其所需学习的逻辑只有单步深度：写入当前所见，并保持此前所写。

回合间传递的记忆经过舍入为运行器实际承接的 `i8` 整数：训练时通过直通估计器（STE，`nets.LearnedMemory`），
在留出集验证时直接作为整数传递，因此此处报告的指标与天梯实际看到的数值完全一致。

## 该指标代表的物理含义

损失函数与 bc.py 相同，即针对每只蚂蚁与教师动作计算交叉熵，使用的数据集也与固定记忆和纯棋盘网络完全一致。
使用带记忆教师的标签时，如果记忆机制学会保留教师后续读取的信息 —— 例如曾见过一次的敌方蚁穴、视野外的食物格子 ——
留出集的一致率将向固定记忆模型靠拢；若什么都没学到，则只能徘徊在纯棋盘网络的水平。
这种在同一数据集上的三组对比实验，正是本脚本存在的意义。训练后请务必实战对弈检验：一致率高并不等同于实战强。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import nets
from ..export import budget, classes, command
from ..planes import ANT_MEMORY, ANT_TABLE, N_MEMORY
from .bc import Rows, ant_logits, device, override, tensors

KEY_RE = re.compile(rb'"k":\[(\d+),(\d+),(\d+)\]')


def sequences(rows: Rows) -> list[list[int]]:
    """按回合顺序整理每个座位的行索引列表；若回合出现断层则开启新序列，因为记忆无法跨断层传递。
    不含 `k` 键的数据集无法进行时序重排。"""
    by_seat: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, line in enumerate(rows.lines):
        m = KEY_RE.search(line)
        if not m:
            raise SystemExit("this dataset's rows carry no `k` (episode, seat, turn): collect it again")
        ep, seat, turn = (int(g) for g in m.groups())
        by_seat[(ep, seat)].append((turn, i))
    out = []
    for items in by_seat.values():
        items.sort()
        run = [items[0][1]]
        for (t0, _), (t1, j) in zip(items, items[1:]):
            if t1 == t0 + 1:
                run.append(j)
            else:
                out.append(run)
                run = [j]
        out.append(run)
    return out


def batches(rows: Rows, seqs: list[list[int]], size: int, rng: random.Random, shuffle: bool = True):
    """同一棋盘尺寸下的序列批次，按长度相邻聚合，以保证批次中的有效座位衰减缓慢。"""
    by_size: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    for s in seqs:
        by_size[rows.sizes[s[0]]].append(s)
    out = []
    for group in by_size.values():
        group.sort(key=len, reverse=True)
        out += [group[i : i + size] for i in range(0, len(group), size)]
    if shuffle:
        rng.shuffle(out)
    return out


def initial_memory(n: int, size, dev) -> torch.Tensor:
    """`n` 个座位在回合 0 的初始记忆：全零张量，无论计算图携带何种类型的记忆。"""
    return torch.zeros((n, N_MEMORY, *size), device=dev)


class AntCarry:
    """单座在回合间传递的 `ant_memory`，在 PyTorch 中按 ID 关联以便梯度反传：
    相当于将适配器的哈希表（`planes._ant_rows_logic`）与散布操作（`ant_planes_adapter`）
    改用 `index_put` 和 `index_select` 重新实现，计算结果完全一致。"""

    def __init__(self, dev, kind: str = "learned"):
        self.dev = dev
        self.kind = kind
        self.rows: torch.Tensor | None = None       # [P, K+1]，按计算图写入的原始格式

    def planes(self, obs: dict) -> torch.Tensor:
        """`[K, rows, cols]`：当前回合蚂蚁继承上一回合对应蚂蚁的值，新生成的蚂蚁或第 0 回合填零，
        处于计算图读取的 u8 尺度。"""
        size = obs["size"]
        out = torch.zeros((ANT_MEMORY, *size), device=self.dev)
        ids = obs.get("ids") or []
        if self.rows is None or not ids:
            return out
        table = torch.zeros((ANT_TABLE, ANT_MEMORY), device=self.dev)
        keys = (self.rows[:, 0].detach().round().long() % ANT_TABLE)
        table = table.index_put((keys,), self.rows[:, 1:].float().clamp(0, 255))
        picked = table.index_select(0, torch.tensor([i % ANT_TABLE for i in ids], device=self.dev))
        mine = torch.tensor(obs["mine"], dtype=torch.long, device=self.dev)
        return out.index_put((torch.arange(ANT_MEMORY, device=self.dev).repeat_interleave(len(ids)),
                              mine[:, 0].repeat(ANT_MEMORY), mine[:, 1].repeat(ANT_MEMORY)),
                             picked.t().reshape(-1))

    @staticmethod
    def inputs(observations: list[dict], dev) -> tuple[torch.Tensor, torch.Tensor]:
        """为一批座位提取 `ids` 与 `cells`，以最宽蚁群为准填充，填充时重复该座位的最后一只蚂蚁
        （重复 gather 不会改变读回的值）。"""
        n = max(1, max(len(o["mine"]) for o in observations))
        ids = torch.zeros((len(observations), n), dtype=torch.long)
        cells = torch.zeros((len(observations), n), dtype=torch.long)
        for i, o in enumerate(observations):
            k = len(o["mine"])
            if k:
                ids[i, :k] = torch.tensor([x % ANT_TABLE for x in o["ids"]])
                ids[i, k:] = ids[i, k - 1]
                cells[i, :k] = torch.tensor([r * o["size"][1] + c for r, c in o["mine"]])
                cells[i, k:] = cells[i, k - 1]
        return ids.to(dev), cells.to(dev)


def run_epoch(model, rows, chunks, dev, opt=None, bptt: int = 32) -> tuple[float, float, int]:
    train = opt is not None
    model.train(train)
    board_memory = getattr(model, "memory_ports", False)
    per_ant = getattr(model, "ant_ports", False)
    ant_kind = getattr(model, "ant_kind", "learned") if per_ant else None
    total_loss = correct = seen = 0
    for seqs in chunks:
        # 最长序列排在最前，使得任何回合活跃的座位均为批次的前缀。
        seqs = sorted(seqs, key=len, reverse=True)
        h = None
        carries = [AntCarry(dev, ant_kind or "learned") for _ in seqs]
        loss_sum, ants = 0.0, 0
        with torch.set_grad_enabled(train):
            for t in range(len(seqs[0])):
                live = [s[t] for s in seqs if t < len(s)]
                boards, _, b, r, c, y = tensors(rows, live, dev)
                parsed = [rows.parse(i) for i in live] if per_ant else None
                args = [boards]
                if board_memory:
                    if h is None:
                        h = initial_memory(len(live), boards.shape[2:], dev)
                    else:
                        h = h[: len(live)]
                    args.append(h)
                if per_ant:
                    observations = [p["o"] for p in parsed]
                    args.append(torch.stack([carries[k].planes(o) for k, o in enumerate(observations)]))
                    args += list(AntCarry.inputs(observations, dev))
                outputs = model(*args)
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                policy, outputs = outputs[0], list(outputs[1:])
                if board_memory:
                    h = outputs.pop(0)
                if per_ant:
                    ant_rows = outputs.pop(0)                       # [live, N, K+1]
                    for k, p in enumerate(parsed):
                        o = p["o"]
                        carries[k].rows = ant_rows[k, : len(o["mine"])]
                if y.numel():
                    logits = ant_logits(policy, b, r, c)
                    loss_sum = loss_sum + F.cross_entropy(logits, y, reduction="sum")
                    ants += y.numel()
                    correct += int((logits.argmax(1) == y).sum())
                last = t + 1 == len(seqs[0])
                if ants and ((t + 1) % bptt == 0 or last):
                    loss = loss_sum / ants
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    total_loss += loss.detach().item() * ants
                    seen += ants
                    loss_sum, ants = 0.0, 0
                    if h is not None:
                        h = h.detach()
                    for carry in carries:
                        if carry.rows is not None:
                            carry.rows = carry.rows.detach()
    return total_loss / max(seen, 1), correct / max(seen, 1), seen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--arch", default=None, help="覆盖该规格的默认网络架构")
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--memory", default="learned",
                    choices=["learned", "max", "none"],
                    help="计算图携带的棋盘记忆类型；`none` 表示仅携带单蚁记忆")
    ap.add_argument("--ants", nargs="?", const="learned", default=False, choices=["learned"],
                    help="额外携带单蚁记忆行（nets.PerAnt）：计算图自主学习的特征值")
    ap.add_argument("--data", type=Path, default=Path("data/teacher-memory.jsonl.gz"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16, help="同步步进的座位数量")
    ap.add_argument("--bptt", type=int, default=8,
                    help="梯度反传回溯的回合数，同时决定优化器的更新频率")
    ap.add_argument("--lr", type=float, default=4e-3)
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
    seqs = sequences(rows)
    rng.shuffle(seqs)
    cut = int(len(seqs) * (1 - a.holdout))
    train_seqs, val_seqs = seqs[:cut], seqs[cut:]

    memory = False if a.memory == "none" else a.memory
    if not memory and not a.ants:
        raise SystemExit("nothing to carry: give a board memory (`--memory learned|max`) or `--ants`")
    spec = override(classes()[a.cls], a)
    model = nets.build(spec, memory=memory, ants=a.ants).to(dev)
    b = budget(a.cls)
    what = " + ".join(([f"{memory} memory"] if memory else [])
                      + (["a memory per ant"] if a.ants else []))
    print(f"{a.cls} + {what}: {nets.policy_params(model):,} parameters "
          f"(budget about {b['params_at_target']:,}), {dev.type}")
    print(f"data: {len(train_seqs):,} sequences train / {len(val_seqs):,} held out over "
          f"{len(rows):,} rows, teacher {header.get('teacher')}, engine {header['engine_digest'][:20]}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    val_chunks = batches(rows, val_seqs, a.batch, rng, shuffle=False)

    out = a.out or Path("runs") / f"{a.cls}-bc-{a.memory if memory else 'no'}-memory{'-' + a.ants if a.ants else ''}"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        chunks = batches(rows, train_seqs, a.batch, rng)
        tl, ta, n = run_epoch(model, rows, chunks, dev, opt, a.bptt)
        vl, va, _ = run_epoch(model, rows, val_chunks, dev, bptt=a.bptt)
        sched.step()
        took = time.time() - t0
        print(f"  epoch {epoch}/{a.epochs}  train {tl:.4f} / {ta:.1%}   "
              f"held out {vl:.4f} / {va:.1%}   {n:,} ants  {took:.0f}s")
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va, "seconds": round(took, 1)})
        if vl < best:
            best = vl
            torch.save({"trunk": model.state_dict(), "class": a.cls, "memory": memory,
                        "ants": a.ants, "engine_digest": header["engine_digest"]}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(
        {"class": a.cls, "method": "bc", "arch": spec["arch"], "spec": spec, "memory": memory,
         "ants": a.ants, "bptt": a.bptt, "batch": a.batch, "data": str(a.data), "teacher": header.get("teacher"),
         "seed": a.seed, "command": command("tb_baselines.train.seq", argv),
         "epochs": history, "engine_digest": header["engine_digest"], "device": dev.type},
        indent=2) + "\n")
    print(f"  -> {out / 'best.pt'}  (best held-out loss {best:.4f})")
    print("  agreement with the teacher is not strength. Play it: `python -m tb_baselines.eval`")


if __name__ == "__main__":
    main()
