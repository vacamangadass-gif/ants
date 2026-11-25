# 运行教师模型并记录其行为：供所有监督学习网络训练的数据集生成器。
# 单个数据集供所有学习器复用，在每行记录原始观测 o、位置 k=[ep, seat, turn]、记忆 m 及教师动作 a。

from __future__ import annotations

import argparse
import gzip
import json
import shlex
import sys
import time
from pathlib import Path

from .env import Env, orders_from_indices
from .planes import MEMORY, remember
from .teacher import CompoundTeacher, TacticalLocalTeacher, Teacher


def collect(
    out: Path,
    seat_turns: int,
    waves: int = 4,
    matches_per_wave: int = 16,
    max_turns: int = 300,
    seed: int = 1,
    maps: str | None = None,
    remembers: bool = False,
    teacher: str = "default",
    hill_lead: int = 6,
    memory_mode: str = "where",
) -> dict:
    """持续运行对局直到写入 `seat_turns` 行数据，并汇报采集统计数据。

    指定 `remembers` 时，教师模型会读取每个座位的记忆（参见 `teacher.py` 中的*会记忆的教师*），
    因此标签会依赖于每行数据在 `m` 键下携带的状态，带记忆的模型也因此有了可供模仿学习的信号。
    """
    if teacher == "default":
        teacher_agent = Teacher(hill_lead=hill_lead)
        teacher_desc = f"deterministic potential field (hill_lead={hill_lead})"
    elif teacher == "compound":
        teacher_agent = CompoundTeacher(hill_lead=hill_lead)
        teacher_desc = "compound situational policy"
    elif teacher == "tactical":
        teacher_agent = TacticalLocalTeacher(hill_lead=hill_lead)
        teacher_desc = f"local tactical focus-battle potential field (hill_lead={hill_lead})"
    else:
        raise ValueError(f"unknown teacher: {teacher!r}; expected 'default', 'compound' or 'tactical'")
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    ants = 0
    episodes = 0
    reasons: dict[str, int] = {}
    scores: list[int] = []
    started = time.time()

    with Env(waves=waves, matches_per_wave=matches_per_wave, max_turns=max_turns,
             seed=seed, maps=maps) as env, gzip.open(out, "wt") as f:
        # 文件头记录了数据集的完整出处。无法说明由哪个引擎生成的数据集是不可复现的，
        # 且引擎的变动属于游戏规则的变动。
        f.write(json.dumps({
            "kind": "teacher-rollout",
            "engine_digest": env.engine_digest,
            # 评估器为 datalogic，环境按版本对其命名，本地通过与远端拒绝之间的版本差异正体现于此。
            "evaluator": env.evaluator,
            "maps": [m["id"] for m in env.hello["maps"]],
            "max_turns": max_turns,
            "env_seed": seed,
            # 教师模型是关于观测数据的纯函数（若开启记忆，则也是前序回合的纯函数，前序回合同样由环境种子唯一决定），
            # 因此环境种子即构成了全部出处：基于该数值与引擎摘要即可完全确定性复现本数据集。
            "teacher": teacher_desc + (", remembering" if remembers else ""),
            "teacher_type": teacher,
            "teacher_remembers": remembers,
            "hill_lead": hill_lead,
            "memory_mode": memory_mode,
            # 每行数据在 `m` 键下携带的记忆状态，按特征平面名称列出。
            "memory": [m.name for m in MEMORY],
        }) + "\n")

        # 记录每个存活座位迄今为止所见的内容，以 episode 和 seat 为键：`ep` 是唯一稳定的键，
        # 因为一个 wave 的 `(w, m)` 在重新填充时会被复用。教师模型仅在明确要求时才读取记忆。
        remembered: dict[tuple[int, int], dict] = {}
        step = env.reset()
        while written < seat_turns:
            actions = []
            for s in step.seats:
                water, seen = _known(s.obs)
                key = (s.ep, s.seat)
                state = remembered.get(key)
                row = {"o": s.obs, "k": [s.ep, s.seat, s.turn]}
                moves = teacher_agent.orders(s.obs, water, seen, state if remembers else None)
                row["m"] = state or {}
                order = orders_from_indices(moves, [len(s.obs["mine"])])[0]
                actions.append(order)
                row["a"] = order
                if written < seat_turns and s.obs["mine"]:
                    # `k` 是该行在整场对局中的位置 (episode, seat, turn)：使训练器能够将座位的行按序重组，
                    # 从而支持计算图自主学习与计算记忆。
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1
                    ants += len(order)
                remembered[key] = remember(state, s.obs, mode=memory_mode)

            step = env.step(actions)
            for e in step.ended:
                episodes += 1
                reasons[e.reason] = reasons.get(e.reason, 0) + 1
                scores.append(max(e.scores))
                for seat in range(e.seat_count):
                    remembered.pop((e.ep, seat), None)
            if not step.seats:
                break

    elapsed = time.time() - started
    return {
        "rows": written,
        "ant_labels": ants,
        "episodes": episodes,
        "reasons": reasons,
        "mean_best_score": round(sum(scores) / len(scores), 2) if scores else None,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(written / max(elapsed, 1e-9)),
        "bytes": out.stat().st_size,
        "teacher": teacher,
        "hill_lead": hill_lead,
        "memory_mode": memory_mode,
    }


def _known(obs):
    from .teacher import known
    return known(obs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="采集教师模型对战推演数据集。")
    ap.add_argument("--out", type=Path, default=Path("data/teacher.jsonl.gz"))
    ap.add_argument("--seat-turns", type=int, default=200_000)
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maps", default=None,
                    help="棋盘 ID、文件路径或包含棋盘的目录；默认为发布包中的基础棋盘")
    ap.add_argument("--remember", action="store_true",
                    help="教师模型读取每个座位的记忆，生成的训练标签将依赖于记忆状态")
    ap.add_argument("--teacher", choices=["default", "compound", "tactical"], default="default",
                    help="教师模型策略类型：'default' 为势能场单一教师，'compound' 为复合调度教师，'tactical' 为局部聚焦战势能场教师（默认 'default'）")
    ap.add_argument("--hill-lead", type=int, default=6,
                    help="敌方蚁丘相对食物的起步领先步数（默认 6，激进进攻设为 12）")
    ap.add_argument("--memory-mode", choices=["where", "max"], default="where",
                    help="记忆更新模式：'where' 为可见性校正条件选择更新，'max' 为历史单调并集累积更新（默认 'where'）")
    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)

    stats = collect(
        a.out, a.seat_turns, waves=a.waves, matches_per_wave=a.matches_per_wave,
        max_turns=a.max_turns, seed=a.seed, maps=a.maps, remembers=a.remember,
        teacher=a.teacher, hill_lead=a.hill_lead, memory_mode=a.memory_mode,
    )
    # 命令行记录在统计数字旁，以便导出脚本在模型卡片上直接打印生成该数据集的配方命令，而非指向 README。
    stats["command"] = shlex.join(["python", "-m", "tb_baselines.collect", *argv])
    (a.out.with_suffix("").with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
