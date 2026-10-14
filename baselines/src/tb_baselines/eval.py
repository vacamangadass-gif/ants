"""通过真实运行路径，让导出的构件两两对弈。

    python -m tb_baselines.eval models/micro-bc models/nano-bc --boards 4

**此处并不直接评估神经网络。** 它会生成比赛定义文件并运行 `tinybrains <match>`，
后者承载卡带环境，通过 datalogic 评估清单中的适配器，通过 tract 运行计算图，
并写入与 Kalam 完全相同的回放包络（replay envelope）——即 Orion 节点链接的相同两个底层库。
因此，这里的对局结果完全是在正式规则约束下产生的结果——包含适配器开销、算力预算限制
以及环面坐标折叠——而不是训练循环中自我验证编码器得出的抽象数字。

这种区别正是本模块设计轻量的根本原因。系统共有三道门禁，且仅有后两道代表真实环境：

    tinybrains env       训练推演              快速；无超时截止，无违规判罚，无适配器
    tinybrains <match>   本模块                真实执行路径，仅略去准入校验
    tinybrains check     export.py             平台准入评测的最终决定依据

## 每对参赛者均在每个棋盘的两个座位上轮流对弈

Ants 棋盘在构造上是对称的，但单场*比赛*并非绝对对称：随机种子决定食物刷新位置，
且在争夺同一格子时其中一个座位拥有优先移动权。
因此，每对选手在每个棋盘上均对弈两次并互换座位，经受住座位互换检验的胜率才是衡量选手实力的真实胜率。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Entrant:
    """代表一个可占据座位的实体：导出的构件目录，或编写的脚本。"""

    name: str
    weights: Path | None = None
    manifest: Path | None = None
    script: list[str] | None = None

    def seat(self, index: int) -> dict:
        s: dict = {"seat": index, "label": self.name}
        if self.script is not None:
            s["script"] = self.script
        else:
            s["weights"] = str(self.weights.resolve())
            s["manifest"] = str(self.manifest.resolve())
        return s


@dataclass
class Record:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score_for: int = 0
    score_against: int = 0
    matches: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.points / self.matches if self.matches else 0.0


def cli() -> str:
    found = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")
    if not found:
        raise SystemExit("no `tinybrains` on PATH; set TINYBRAINS to the binary")
    return found


def boards(maps: str | None, limit: int) -> list[str]:
    """获取待对战的双人棋盘列表，按比赛文件中的命名方式表示：棋盘 ID 或文件路径。

    `maps` 可以是棋盘所在目录（如某个赛季的棋盘集），或者是逗号分隔的 ID/路径列表；
    若为 None，则直接从 `tinybrains maps` 读取发布版本自带的所有双人棋盘，而非硬编码，
    因为棋盘列表归卡带所有。这意味着需要解析人类可读的表格文本，因此解析过程带有校验（CHECKED）：
    若结果为空，否则会导致不进行任何比赛并打印全为 0 的循环赛表格，这看起来像全盘平局而非 Bug。
    循环赛是双人对抗，因此座位数大于 2 的棋盘不会包含在此列表中。
    """
    if maps and Path(maps).is_dir():
        picked = []
        for f in sorted(Path(maps).glob("*.json")):
            if json.loads(f.read_text()).get("players") == 2:
                picked.append(str(f.resolve()))
        if not picked:
            raise SystemExit(f"no two-seat boards in {maps}")
        return picked[:limit]
    if maps:
        return [m.strip() for m in maps.split(",") if m.strip()][:limit]

    out = subprocess.run([cli(), "maps"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"tinybrains maps failed:\n{out.stderr or out.stdout}")
    rows = []
    for line in out.stdout.splitlines()[1:]:      # 第 0 行为 "<game> N boards" 表头
        parts = line.split()
        # id、尺寸、座位数 —— 结构特征足以在表格格式变动时感知。
        if len(parts) >= 4 and "x" in parts[1] and parts[2] == "2" and parts[3] == "seats":
            rows.append(parts[0])
    if not rows:
        raise SystemExit(
            "could not read a two-seat board out of `tinybrains maps`; its output format has "
            "changed, or the release ships none:\n" + "\n".join(out.stdout.splitlines()[:4])
        )
    return rows[:limit]


def play(entrants: list[Entrant], board_limit: int, maps: str | None,
         max_turns: int, seed: int, out_dir: Path) -> dict[str, Record]:
    """每对选手、双方互换座位、在每个棋盘上对战 —— 生成单个比赛批次文件，
    因为一个波次（wave）需要统一的座位数，而此处所有棋盘均为双人座。"""
    records = {e.name: Record() for e in entrants}
    pairs = list(itertools.combinations(range(len(entrants)), 2))
    if not pairs:
        raise SystemExit("a round robin needs at least two entrants")

    rows: list[dict] = []
    labels: dict[str, tuple[str, str]] = {}
    for board in boards(maps, board_limit):
        name = Path(board).stem if board.endswith(".json") else board
        for a, b in pairs:
            for flip in (0, 1):
                first, second = (a, b) if not flip else (b, a)
                mid = f"{entrants[first].name}-vs-{entrants[second].name}-{name}-{flip}"
                rows.append({
                    "id": mid,
                    "seed": seed + len(rows),
                    "map": board,
                    "seat_count": 2,
                    "seats": [entrants[first].seat(0), entrants[second].seat(1)],
                })
                labels[mid] = (entrants[first].name, entrants[second].name)

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mf = Path(tmp) / "round-robin.json"
        mf.write_text(json.dumps(
            {"game": "ants", "vars": {"max_turns": max_turns}, "rows": rows}))
        r = subprocess.run(
            [cli(), str(mf), "--out", str(out_dir)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"round robin: {r.stderr or r.stdout}")

        for row in rows:
            replay = json.loads((out_dir / f"{row['id']}.json").read_text())
            left, right = labels[row["id"]]
            ranks, scores = replay["engine_ranks"], replay["scores"]
            for who, mine, theirs in ((left, 0, 1), (right, 1, 0)):
                rec = records[who]
                rec.matches += 1
                rec.score_for += scores[mine]
                rec.score_against += scores[theirs]
                rec.reasons[replay["reason"]] = rec.reasons.get(replay["reason"], 0) + 1
                if ranks[mine] < ranks[theirs]:
                    rec.wins += 1
                elif ranks[mine] > ranks[theirs]:
                    rec.losses += 1
                else:
                    rec.draws += 1
    return records


def table(records: dict[str, Record]) -> str:
    lines = [f"{'entrant':22s} {'played':>7s} {'W':>4s} {'D':>4s} {'L':>4s} "
             f"{'rate':>6s} {'score':>7s} {'against':>8s}"]
    for name, r in sorted(records.items(), key=lambda kv: -kv[1].win_rate):
        lines.append(
            f"{name:22s} {r.matches:>7d} {r.wins:>4d} {r.draws:>4d} {r.losses:>4d} "
            f"{r.win_rate:>5.0%} {r.score_for / max(r.matches, 1):>7.2f} "
            f"{r.score_against / max(r.matches, 1):>8.2f}"
        )
    return "\n".join(lines)


def entrant_from(path: str) -> Entrant:
    p = Path(path)
    if p.is_dir():
        return Entrant(name=p.name, weights=p / "model.onnx", manifest=p / "manifest.json")
    raise SystemExit(f"{path} is not an exported model directory")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrants", nargs="+", help="导出的模型目录")
    ap.add_argument("--boards", type=int, default=3, help="最多对战的双人棋盘数量")
    ap.add_argument("--maps", default=None,
                    help="棋盘 ID、路径或棋盘所在目录；默认使用发布版本自带的棋盘")
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--out", type=Path, default=Path("replays"))
    ap.add_argument("--json", dest="as_json", action="store_true")
    a = ap.parse_args(argv)

    entrants = [entrant_from(e) for e in a.entrants]
    records = play(entrants, a.boards, a.maps, a.max_turns, a.seed, a.out)
    if a.as_json:
        print(json.dumps({n: vars(r) for n, r in records.items()}, indent=2))
    else:
        print(table(records))
        print()
        print("Played under the rules -- the manifest, the operation budget and the wrap included.")
        print("It is not the ladder: no ratings, no seasons, and no trial.")


if __name__ == "__main__":
    main()
