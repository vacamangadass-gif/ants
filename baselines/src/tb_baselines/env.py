"""`tinybrains env` 的客户端：将游戏卡带作为批处理训练环境使用。

单子进程通信，双向通过 JSON Lines 交互，并且依靠一条确保安全的核心规则：**动作指令与上一次调用返回的存活座位按位置一一对齐**。这是游戏通信协议本身的规则，在训练环境中同样成立，因为训练环境绝不会强制某座位弃权，因此每个存活座位在每个回合都会被执行。

    env = Env(waves=4, matches_per_wave=16, max_turns=300, seed=1)
    step = env.reset()
    while True:
        orders = my_policy(step.boards)          # [live, planes, rows, cols] -> [live] 字符串列表
        step = env.step(orders)

`Step.boards` 已经由 `planes.encode` 完成特征编码 —— 该函数在 `tests/test_adapter_conformance.py` 中被证明与 `manifest.json` 完全等价。`Step.observations` 是并排保留的原始 JSON 观测数据，因为奖励信号由游戏规则计算而非张量计算。

## 本模块不是什么

它不是正式裁判。这里没有回合时间限制，没有犯规累计上限，没有弃权判负，也没有适配器评估开销，因此在训练环境中运行良好的策略依然可能在真实对弈中失败。`tinybrains check` 与真实的 `tinybrains <match>` 才是真正的检验门禁；详见 `eval.py`。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .planes import MOVES, encode_batch

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Seat:
    """本回合存活的一个座位。`ep` 是唯一稳定的对局标识：`(w, m)` 在重新填充时会被复用。"""

    ep: int
    seat: int
    wave: int
    match: int
    turn: int
    obs: dict


@dataclass
class Ended:
    ep: int
    seat_count: int
    ranks: list[int]
    scores: list[int]
    reason: str
    turns: int
    map: str                                # 进行比赛的棋盘文件内容或标识
    map_id: str
    seed: int


@dataclass
class Group:
    """同一棋盘尺寸下的所有存活座位，堆叠为一个批次张量。

    **批次是按棋盘尺寸划分的，这并非实现细节。** 缓冲池容纳了多个 wave，每个 wave 在单张棋盘上进行，
    而棋盘具有多种尺寸 —— 5 张基础棋盘从 24x24 到 120x124，赛季棋盘则可为该范围内的任意尺寸。策略网络是全卷积网络，
    因此可在任意尺寸上运行，但单个张量无法同时容纳不同尺寸，所以训练器会按组遍历。在天梯上绝无此问题：每场比赛的每个座位
    都是独立的 `model_infer` 调用。
    """

    size: tuple[int, int]
    indices: list[int]                      # 对应 `Step.seats` 中的索引
    boards: np.ndarray                      # 形状为 [n, planes, rows, cols]，类型为 int8


@dataclass
class Step:
    seats: list[Seat]
    groups: list[Group]
    scores: dict[int, list[int]]            # ep -> 当前每个座位的得分列表
    ended: list[Ended]
    turn: int
    observations: list[dict] = field(default_factory=list)

    @property
    def boards(self) -> np.ndarray:
        """单一尺寸组的棋盘张量，适用于锁定到单一尺寸棋盘的运行。当缓冲池包含多种尺寸时会主动报错，
        而不是静默返回残缺的部分批次。"""
        if len(self.groups) != 1:
            raise EnvError(
                f"当前 step 跨越了 {len(self.groups)} 种棋盘尺寸 "
                f"({', '.join(f'{r}x{c}' for r, c in (g.size for g in self.groups))})。\n"
                "请循环遍历 `step.groups`，或向 Env 传入单一尺寸的 `maps=` 参数以锁定棋盘尺寸。"
            )
        return self.groups[0].boards


class EnvError(RuntimeError):
    pass


class Env:
    """子进程、通信协议及核心封装。

    确定性：给定 `seed` 随机种子与动作流，每个棋盘、每次食物刷新以及每场比赛均完全可复现。
    这使得训练运行成为完全可重复的过程。

    `maps` 为棋盘池：官方发布的棋盘 ID、棋盘文件路径，或包含棋盘的目录（例如某个赛季的棋盘目录）——
    可为列表或逗号分隔的字符串。为 None 时默认采用发布包的基础棋盘。
    没有预设棋盘：引擎自身不携带任何棋盘，因此棋盘池直接由棋盘数据定义。
    """

    def __init__(
        self,
        waves: int = 4,
        matches_per_wave: int = 16,
        max_turns: int = 300,
        seed: int = 1,
        maps: str | list[str] | None = None,
        game: str | None = None,
        scores: str = "every",
        cwd: Path | None = None,
    ):
        binary = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")
        if not binary:
            raise EnvError(
                "PATH 中未找到 `tinybrains` 二进制文件。\n"
                "  brew tap tiny-brains/cli https://github.com/Tiny-Brains/cli\n"
                "  brew install tiny-brains/cli/tinybrains   (或将 TINYBRAINS 设置为该可执行文件路径)"
            )
        argv = [
            binary, "env",
            "--waves", str(waves),
            "--matches-per-wave", str(matches_per_wave),
            "--max-turns", str(max_turns),
            "--seed", str(seed),
            "--scores", scores,
        ]
        if maps:
            argv += ["--maps", maps if isinstance(maps, str) else ",".join(str(m) for m in maps)]
        if game:
            argv += ["--game", game]

        self.proc = subprocess.Popen(
            argv, cwd=str(cwd or ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.hello = self._read()["hello"]
        self.closed = False
        # 每个模型卡片都必须注明这些信息。无法说明由哪个引擎生成数据的运行是无法复现的。
        self.engine_digest = self.hello["engine_digest"]
        # 现在的评估器是 datalogic，按版本而非摘要命名：适配器由节点上的同一代码库计费并执行，
        # 因此版本偏差是导致本地通过而远端拒绝的原因。
        self.evaluator = self.hello["evaluator"]

    # ---- 通信传输端 ----------------------------------------------------------------

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise EnvError("环境子进程已退出")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise EnvError(reply.get("error", "环境拒绝了请求"))
        return reply

    def _write(self, request: dict) -> None:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

    def _pack(self, reply: dict) -> Step:
        seats = [
            Seat(ep=s["ep"], seat=s["seat"], wave=s["w"], match=s["m"], turn=s["turn"], obs=s["obs"])
            for s in reply["seats"]
        ]
        obs = [s.obs for s in seats]
        by_size: dict[tuple[int, int], list[int]] = {}
        for i, o in enumerate(obs):
            by_size.setdefault(tuple(o["size"]), []).append(i)
        groups = [
            Group(size=size, indices=idx, boards=encode_batch([obs[i] for i in idx]))
            for size, idx in sorted(by_size.items())
        ]
        return Step(
            seats=seats,
            groups=groups,
            scores={r["ep"]: r["scores"] for r in reply["scores"]},
            ended=[
                Ended(
                    ep=e["ep"], seat_count=len(e["scores"]), ranks=e["ranks"], scores=e["scores"],
                    reason=e["reason"], turns=e["turns"], map=e["map"],
                    map_id=e["map_id"], seed=e["seed"],
                )
                for e in reply["ended"]
            ],
            turn=reply["turn"],
            observations=obs,
        )

    # ---- 交互循环 ------------------------------------------------------------------

    def reset(self) -> Step:
        self._write({"op": "observe"})
        return self._pack(self._read())

    def step(self, actions: list[str]) -> Step:
        """`actions[i]` 对应 `seats[i]`：每只蚂蚁对应一个字符，按 `mine` 的顺序排列。

        该紧凑格式是回放系统自身的原生格式（参见 `ants/engine/src/replay.rs`），而非发明出来的格式，
        它在网络传输上的体积比字符串数组小 4 倍。
        """
        self._write({"op": "step", "actions": actions})
        return self._pack(self._read())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._write({"op": "close"})
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def orders_from_indices(indices: np.ndarray, counts: list[int]) -> list[str]:
    """将一维移动动作索引数组按座位切分，转为环境所需的紧凑字符串列表。

    `counts[i]` 是座位 `i` 拥有的蚂蚁数量，即 `len(obs["mine"])` —— 动作数组按位置
    与 `mine` 严格对齐，没有任何其他机制寻址具体蚂蚁。
    """
    table = np.array([ord(m) for m in MOVES], dtype=np.uint8)
    chars = table[np.clip(indices, 0, len(MOVES) - 1)]
    out, at = [], 0
    for n in counts:
        out.append(chars[at : at + n].tobytes().decode("ascii"))
        at += n
    return out
