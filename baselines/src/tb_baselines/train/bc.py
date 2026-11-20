# 行为克隆（Behaviour Cloning）：将指定规格等级的网络拟合到教师策略动作上。
# 损失函数针对单只蚂蚁计算交叉熵，支持 Where 记忆平面、多线程预取与 CUDA bfloat16 混合精度加速。

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


def _prepare_cpu_batch(rows: Rows, batch: list[int], memory: bool = False):
    parsed = [rows.parse(i) for i in batch]
    boards = np.concatenate([encode(r["o"]) for r in parsed], axis=0)
    mem_np = (
        np.concatenate([encode_memory(r.get("m"), r["o"]["size"]) for r in parsed], axis=0)
        if memory
        else None
    )
    b_idx, rr, cc, yy = [], [], [], []
    for i, r in enumerate(parsed):
        for (row, col), ch in zip(r["o"]["mine"], r["a"]):
            b_idx.append(i)
            rr.append(row)
            cc.append(col)
            yy.append(MOVE_INDEX.get(ch, MOVE_INDEX["-"]))
    return (
        torch.from_numpy(boards),
        torch.from_numpy(mem_np) if mem_np is not None else None,
        torch.tensor(b_idx, dtype=torch.long),
        torch.tensor(rr, dtype=torch.long),
        torch.tensor(cc, dtype=torch.long),
        torch.tensor(yy, dtype=torch.long),
    )


def tensors(rows: Rows, batch: list[int], dev: torch.device, memory: bool = False):
    """将一个批次转换为 `(boards, memory, sample, row, col, label)`，在其中所有蚂蚁上展开展平。
    除非显式要求，否则 `memory` 为 None：各行对应座位记录的记忆状态，作为计算图的 `memory_in`。"""
    boards_t, mem_t, b_t, r_t, c_t, y_t = _prepare_cpu_batch(rows, batch, memory)
    return (
        boards_t.to(dev),
        mem_t.to(dev) if mem_t is not None else None,
        b_t.to(dev),
        r_t.to(dev),
        c_t.to(dev),
        y_t.to(dev),
    )


def ant_logits(policy: torch.Tensor, b, r, c) -> torch.Tensor:
    """将 `[batch, moves, rows, cols]` 在有蚂蚁的单元格处下投影为 `[ants, moves]`。

    这与适配器的 `out` 程序通过 `tb.gather` 执行的投影完全相同，这也正是重点所在：
    此处训练的优化目标，正是实际对局中决定移动意图的物理量。
    """
    return policy[b, :, r, c]


def _prefetched_batches(rows: Rows, chunks: list[list[int]], memory: bool, workers: int = 4, prefetch: int = 8):
    if not chunks:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        q = deque()
        it = iter(chunks)
        for _ in range(min(prefetch, len(chunks))):
            b = next(it, None)
            if b is not None:
                q.append(pool.submit(_prepare_cpu_batch, rows, b, memory))
        while q:
            fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append(pool.submit(_prepare_cpu_batch, rows, nxt, memory))
            yield fut.result()


def run_epoch(model, rows, chunks, dev, opt=None, memory: bool = False) -> tuple[float, float, int]:
    train = opt is not None
    model.train(train)
    total_loss_t = torch.zeros((), device=dev, dtype=torch.float64)
    correct_t = torch.zeros((), device=dev, dtype=torch.int64)
    seen = 0
    use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    for boards_cpu, mem_cpu, b_cpu, r_cpu, c_cpu, y_cpu in _prefetched_batches(rows, chunks, memory):
        if y_cpu.numel() == 0:
            continue
        boards = boards_cpu.to(dev, non_blocking=True)
        mem = mem_cpu.to(dev, non_blocking=True) if mem_cpu is not None else None
        b = b_cpu.to(dev, non_blocking=True)
        r = r_cpu.to(dev, non_blocking=True)
        c = c_cpu.to(dev, non_blocking=True)
        y = y_cpu.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # 带记忆模型返回 (policy, memory)；记忆是确定性函数，不参与梯度反传。
                policy = model(boards, mem)[0] if memory else model(boards)
                logits = ant_logits(policy, b, r, c).float()
                loss = F.cross_entropy(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # 存活蚂蚁数在 1 到 180 之间浮动，因此批次的梯度尺度也随之大幅波动。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        n = y.numel()
        total_loss_t += loss.detach().double() * n
        correct_t += (logits.argmax(1) == y).sum()
        seen += n
    if seen == 0:
        return 0.0, 0.0, 0
    return float(total_loss_t.item()) / seen, int(correct_t.item()) / seen, seen


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
    ap = argparse.ArgumentParser(description="将指定规格等级的网络拟合到教师策略动作的行为克隆训练器。")
    ap.add_argument("--class", dest="cls", required=True,
                    help="规格等级名称，如 nano, micro 等（见 classes.toml）")
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
    ap.add_argument("--init-weights", type=Path, default=None, help="可选初始检查点权重路径，用于 DAgger 迭代间热启动")
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
    if a.init_weights is not None:
        if not a.init_weights.exists():
            raise FileNotFoundError(f"指定的初始检查点权重文件不存在: {a.init_weights}")
        ckpt = torch.load(a.init_weights, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["trunk"])
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
