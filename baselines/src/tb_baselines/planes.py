"""观测特征编码：一套规范，双重渲染实现。

**本模块的存在是为了确保训练与推理服务间的偏差（train/serve skew）绝不可能被隐蔽发生。** 在 Python 中进行训练的参赛者需要将每个观测编码两次 —— 一次在 `manifest.json` 中（供天梯平台运行），一次在 NumPy 中（供优化器读取）。同一套编码的两套独立实现是导致模型在竞技场中的得分远低于训练表现的经典根源，且这种失败是静默的：两部分代码都能跑通，仅仅是它们得出的特征不一致。

因此，所有特征平面在此统一定义，且每个定义都并排包含**两套**实现：进入 `manifest.json` 的 JSONLogic 代码片段，以及训练器调用的 NumPy 函数。随后，`tests/test_adapter_conformance.py` 会在卡带的官方参考观测数据上运行真正的评估器 —— `tinybrains adapt`（即 Orion 节点执行适配器所使用的 datalogic 评估器），并断言两者逐元素严格完全一致。两套实现紧邻放置使其易于同步维护；自动化测试则对此提供严格证明。

## 特征平面（The planes）

其中 6 个来自基准参考适配器且保持不变，因为它们经过了充分验证且计算廉价。第 7 个是可见性掩码（visibility mask），**由游戏引擎直接发送**：节点评估适配器所使用的表达式语言无法引用外层迭代器的元素，因此针对每只蚂蚁计算圆盘只能要么展开为 241 次的硬编码卷积核，要么根本无法实现。视野半径是游戏规则的一部分，因此卡带只计算一次，`vis` 随 `water` 一道发送（参见参赛指南中的 *What your model sees*）。该平面与其他平面一样采用 `rle_expand` 解压。

归属者阵营相对于观察者标记 —— 自己永远是 0 号 —— 这使得 `hills` 可以被明确拆分。该机制在引擎 `sha256:f17b51b6c92b` 中被固化；在更旧的引擎版本下，每场比赛 1 号座位的这两个平面会被反转。

## 记忆机制（The memory）

每个座位可以跨回合传递记忆（参见手册的 *Memory* 页面）：计算图输出一个张量，运行器在下一回合该座位的视图中原样传回该键。此处的实现是最基础的形式：视图在回合间遗忘的两个平面，通过 `Max` 算子维护，因此一旦某个单元格上出现过食物或敌方蚁穴，其读数即为 1。它是座位历史所见内容的固定函数，其本身不包含任何可学习参数；读取它的策略网络才是学习的对象。`MEMORY` 声明了这一结构，并由三套实现渲染，`tests/test_adapter_conformance.py` 严格保证它们的一致性：

- 适配器端（`memory_adapter`）：传递上一回合的记忆，第 0 回合生成全 0
- 计算图端（`nets.WithMemory`）：`memory = Max(memory_in, board[sources])`
- NumPy 端（`remember`, `encode_memory`）：迄今为止所见单元格的并集平面
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# 观测自身使用的词汇定义，重命名时只需修改一处而非全文搜索。
VIEW_RADIUS2 = 77

# 数据值的数据类型拼写，对应清单中的声明及张量算子接收的格式。旧方言曾拼写为 `int8`。
DTYPE = "i8"
NP_DTYPE = np.int8


@dataclass(frozen=True)
class Plane:
    """棋盘张量的一个特征通道平面。

    `logic` 是在 `manifest.json` 内部计算该平面的 JSONLogic 表达式，
    作为尺寸表达式的函数（适配器不能硬编码棋盘尺寸：一个赛季包含多种棋盘尺寸）。
    `numpy` 针对训练器从同一观测数据计算完全相同的平面。
    """

    name: str
    why: str
    logic: Callable[[Any], Any]
    numpy: Callable[[dict, "Board"], np.ndarray]


_DISKS: dict[int, np.ndarray] = {}


def _disk(radius2: int) -> np.ndarray:
    """欧氏距离平方 radius² 内的偏移量，每个半径仅计算一次并缓存。

    全程采用纯整数运算，且在半径平方上使用 `<=`，因为这正是 `tb.dilate` 所测试的规则 ——
    若采用浮点半径加 `<`，则恰好会在边界处丢失一圈单元格，而在半径平方 77 时，这占了掩码可观的比例。
    """
    if radius2 not in _DISKS:
        r = int(np.floor(np.sqrt(radius2)))
        _DISKS[radius2] = np.array(
            [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
             if dr * dr + dc * dc <= radius2],
            dtype=np.int64,
        )
    return _DISKS[radius2]


class Board:
    """NumPy 渲染实现的临时工作空间：记录棋盘尺寸，按需提供全零平面。"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

    def zeros(self) -> np.ndarray:
        return np.zeros((self.rows, self.cols), dtype=NP_DTYPE)

    def scatter(self, points) -> np.ndarray:
        """将散点绘制到平面上。越界点会被丢弃而非报错，这与 datalogic 的 `scatter` 行为一致：
        “通常的生成源是一个检测器，在源坐标系中输出边界框，这些框可能会落在目标网格之外”。"""
        g = self.zeros()
        if len(points) == 0:
            return g
        p = np.asarray(points, dtype=np.int64)
        r, c = p[:, 0], p[:, 1]
        keep = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        g[r[keep], c[keep]] = 1
        return g

    def rle(self, runs) -> np.ndarray:
        """行优先游程编码 `[v0, n0, v1, n1, …]`，与 `rle_expand` 读取格式一致。"""
        flat = np.zeros(self.rows * self.cols, dtype=NP_DTYPE)
        at = 0
        for i in range(0, len(runs) - 1, 2):
            v, n = int(runs[i]), int(runs[i + 1])
            if v:
                flat[at : at + n] = 1
            at += n
        return flat.reshape(self.rows, self.cols)

    def dilate(self, g: np.ndarray, radius2: int) -> np.ndarray:
        """在环面上将非零单元格在欧氏半径平方 radius² 内的所有单元格置为 1。

        **不再属于特征编码的计算环节**：`vis` 直接包含在观测中。之所以保留此函数，是因为它是独立的参照 ——
        `tests/test_adapter_conformance.py` 会断言引擎发送的掩码与本函数的计算结果完全相等，从而确保信任引擎发送的数据是安全的。

        **从非零单元格向外发散，而非在整个平面上循环滚动（roll）。** 该算子最直观的写法是“将整个棋盘按圆盘内的每个偏移量平移并做 OR 运算”，
        过去确实曾这样实现：在半径平方 77 下有 241 个偏移量，每个执行两次 `np.roll`，跨越 16,384 个单元格 ——
        在两分钟的性能分析中累计执行了 186 万次 roll 调用，占了整个训练循环耗时的 48%。

        等价形式是从每个已标记的非零单元格向外遍历圆盘。根据数学定义结果完全一致，而计算成本从整个棋盘的尺寸降低为蚂蚁的数量：
        36 只蚂蚁乘以 241 个偏移量仅为 8,700 次写入，对比原先的 390 万次。`tests/test_adapter_conformance.py` 负责保证这种“定义上的等价”绝对诚实可靠。
        """
        out = self.zeros()
        points = np.argwhere(g != 0)
        if len(points) == 0:
            return out
        disk = _disk(radius2)
        r = (points[:, None, 0] + disk[None, :, 0]) % self.rows
        c = (points[:, None, 1] + disk[None, :, 1]) % self.cols
        out[r.ravel(), c.ravel()] = 1
        return out


# ---- JSONLogic 逻辑端 ------------------------------------------------------------------

def var(path: str) -> dict:
    return {"var": path}


SIZE = var("size")


def _scatter(points) -> Callable[[Any], Any]:
    return lambda size: {"scatter": [points, size, DTYPE]}


def _rc_only(source) -> dict:
    """将 `[r, c, owner]` 三元组裁剪为 `[r, c]` 坐标对：`scatter` 两者均可接收，但第三个元素会被视为
    写入的*数值*，若将所有者 ID 作为数值写入就无法构成二值掩码。"""
    return {"map": [source, [var("0"), var("1")]]}


def _hills(mine: bool) -> Callable[[Any], Any]:
    test = {"==": [var("2"), 0]} if mine else {"!=": [var("2"), 0]}
    return lambda size: {
        "scatter": [_rc_only({"filter": [var("hills"), test]}), size, DTYPE]
    }


PLANES: tuple[Plane, ...] = (
    Plane(
        "mine",
        "己方蚂蚁；无论有无战争迷雾，全图唯一完整告知的位置",
        _scatter(var("mine")),
        lambda o, b: b.scatter(o["mine"]),
    ),
    Plane(
        "foes",
        "本回合视野内可见的敌方蚂蚁 —— 从不记忆，因此该平面会随移动而闪烁",
        lambda size: {"scatter": [_rc_only(var("foes")), size, DTYPE]},
        lambda o, b: b.scatter([f[:2] for f in o["foes"]]),
    ),
    Plane(
        "food",
        "视野内可见的食物；经济体系的核心，因为蚂蚁由食物转化而来",
        _scatter(var("food")),
        lambda o, b: b.scatter(o["food"]),
    ),
    Plane(
        "water",
        "已知水域：唯一自带记忆的字段，也是你唯一能逐步积累的全局地图",
        lambda size: {"rle_expand": [var("water.rle"), size, DTYPE]},
        lambda o, b: b.rle(o["water"]["rle"]),
    ),
    Plane(
        "hill_mine",
        "己方蚁穴（归属者 0）—— 被摧毁时每个扣除 1 分",
        _hills(mine=True),
        lambda o, b: b.scatter([h[:2] for h in o["hills"] if h[2] == 0]),
    ),
    Plane(
        "hill_foe",
        "敌方蚁穴 —— 摧毁每个可获得 2 分",
        _hills(mine=False),
        lambda o, b: b.scatter([h[:2] for h in o["hills"] if h[2] != 0]),
    ),
    Plane(
        "visible",
        "此刻当前能够看到的视野，使得 `water` 中的 0 不再同时混淆“已知无障碍”与“从未探索”。"
        "由引擎直接发送 —— 视野半径属于游戏规则，适配器自身无法高效计算圆盘并集。",
        lambda size: {"rle_expand": [var("vis.rle"), size, DTYPE]},
        lambda o, b: b.rle(o["vis"]["rle"]),
    ),
)

N_PLANES = len(PLANES)

# 五个移动方向，按策略网络通道对应的含义排序。`-` 表示原地驻留，排在最后，
# 因为如果最后一通道在所有位置都占优，模型看起来像是自主选择消极防守，实则不然 ——
# 手册中的 *Testing* 章节展示了那种原地不动的蚁群。
MOVES = ("N", "E", "S", "W", "-")
N_MOVES = len(MOVES)


def encode(obs: dict) -> np.ndarray:
    """将单次观测编码为 `[1, planes, rows, cols]`，与清单适配器的产出完全一致。

    开头的 1 是 batch 批次维度，计算图声明其为动态维度，使得训练器能够使用与天梯单座推理相同的计算图堆叠批次数据。
    """
    rows, cols = obs["size"]
    b = Board(rows, cols)
    planes = np.stack([p.numpy(obs, b) for p in PLANES], axis=0)
    return planes.reshape(1, N_PLANES, rows, cols)


def encode_batch(observations: list[dict]) -> np.ndarray:
    """批量编码一个 wave 的观测数据。一个 wave 中的所有对局都在同一张棋盘上进行，
    因此尺寸完全相同 —— 环境每次给 `worldgen` 传入单张棋盘 —— 所以这些数据总能安全堆叠。"""
    return np.concatenate([encode(o) for o in observations], axis=0)


# ---- 记忆机制 ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryPlane:
    """记忆张量的一个平面：棋盘平面的 1 值集合，在比赛剩余时间内持续保留。"""

    name: str
    why: str
    source: str


MEMORY: tuple[MemoryPlane, ...] = (
    MemoryPlane(
        "food_seen",
        "曾经出现过食物的所有单元格：地图从固定点位向蚁群持续补给资源",
        "food",
    ),
    MemoryPlane(
        "hill_foe_seen",
        "曾经见过的所有敌方蚁穴，视野在没有蚂蚁靠近时会在下一回合立即遗忘它",
        "hill_foe",
    ),
)
N_MEMORY = len(MEMORY)
# `PLANES` 中的通道索引，计算图在更新记忆时从这些通道中进行 gather 提取。
MEMORY_SOURCES: tuple[int, ...] = tuple(
    next(i for i, p in enumerate(PLANES) if p.name == m.source) for m in MEMORY
)
# `PLANES` 中可见性掩码（visible）平面的通道索引，用于条件选择更新记忆。
VIS_SOURCE: int = 6
assert PLANES[VIS_SOURCE].name == "visible", f"VIS_SOURCE {VIS_SOURCE} must point to 'visible'"


def memory_adapter(size) -> dict:
    """`memory_in` 适配器：计算图在上一回合写入的记忆，第 0 回合为全零。

    `{"tensor": [{"var": "memory"}]}` 将传递的张量按照输出声明的数据类型和形状原样传递，
    在计算节点和在 `tinybrains` 中完全一致；适配器本身从不窥探其内部结构。
    """
    return {
        "if": [
            var("memory"),
            {"tensor": [var("memory")]},
            {"zeros": [{"merge": [[1, N_MEMORY], size]}, DTYPE]},
        ]
    }


def remember(state: dict | None, obs: dict, mode: str = "where") -> dict:
    """计入当前观测后一个座位所见的内容：依据更新模式融合历史记忆与本回合观测。

    `state` 格式为 `{name: [[r, c], ...]}`，经排序且无重复项 —— 与 `collect.py` 写入数据集行的格式相同 ——
    在第 0 回合为 None。

    更新模式 `mode`：
    - "where"（默认，IDEA-002 可见性校正）：
      提取当前可见性掩码 V_t。对于当前视野内（V_t == 1）的单元格，以当前观测 O_t 覆盖；
      对于迷雾遮挡（V_t == 0）的单元格，保留历史记忆 M_t。
      从而在敌方蚁丘被摧毁进入视野后彻底消除残留，杜绝“幽灵敌丘”导致的无效折返。
      与计算图端 `torch.where(vis, seen, memory_in)` 逐位严格等价。
    - "max"（基准/消融实验，EXP-010 / EXP-012）：
      保留原始单调并集累积 M_{t+1} = M_t ∪ O_t，等价于 `torch.maximum(memory_in, seen)`。
    """
    rows, cols = obs["size"]
    b = Board(rows, cols)
    out = {}
    if mode == "where":
        vis = PLANES[VIS_SOURCE].numpy(obs, b)
        for m, k in zip(MEMORY, MEMORY_SOURCES):
            curr_pts = {(int(r), int(c)) for r, c in np.argwhere(PLANES[k].numpy(obs, b) != 0)}
            prev_pts = {
                (int(r), int(c))
                for r, c in (state or {}).get(m.name, [])
                if 0 <= r < rows and 0 <= c < cols and vis[r, c] == 0
            }
            out[m.name] = [list(p) for p in sorted(prev_pts | curr_pts)]
    elif mode == "max":
        for m, k in zip(MEMORY, MEMORY_SOURCES):
            seen = {tuple(p) for p in (state or {}).get(m.name, [])}
            seen |= {(int(r), int(c)) for r, c in np.argwhere(PLANES[k].numpy(obs, b) != 0)}
            out[m.name] = [list(p) for p in sorted(seen)]
    else:
        raise ValueError(f"unknown memory mode: {mode!r}; expected 'where' or 'max'")
    return out


def encode_memory(state: dict | None, size) -> np.ndarray:
    """将记住的状态转换为 `[1, N_MEMORY, rows, cols]`：即计算图的 `memory_in` 输入。"""
    rows, cols = size
    b = Board(rows, cols)
    planes = np.stack([b.scatter((state or {}).get(m.name, [])) for m in MEMORY], axis=0)
    return planes.reshape(1, N_MEMORY, rows, cols)


def memory_in_view(obs: dict) -> np.ndarray:
    """`memory_adapter` 的 NumPy 实现：视图携带的记忆张量，若未携带则返回全零张量。"""
    if obs.get("memory") is None:
        return encode_memory(None, obs["size"])
    return np.asarray(obs["memory"], dtype=NP_DTYPE).reshape(1, N_MEMORY, *obs["size"])


# ---- 单蚁记忆 ---------------------------------------------------------------------------
#
# `ant_memory` 是每只蚂蚁对应的一行数据，格式为 `[id, s_1 .. s_K]`（数据类型为 `u8`），由计算图在每回合写入，
# 运行器在下一回合的视图中传回（参见手册的 *Memory* 页面，*A memory per ant* 小节）。这些数据行按 ID 严格
# 追踪蚂蚁，绝不按坐标位置追踪：`mine` 在每回合都会重新排序，因此适配器将上一回合的数据行转换为以 ID 为索引的表，
# 并在本回合按照 `ids` 顺序读回。状态值 `s_k` 由计算图自行学习更新（`nets.PerAnt`）；ID 来自视图本身，
# 对 `ANT_TABLE` 取模以确保能够容纳在一个字节内。
#
# 该表会发生回绕，新蚂蚁无法稳定读取到全零。蚂蚁 ID 是按座位累计递增的，因此很快就会超过 `ANT_TABLE`：
# 在 `viz/last-frame-basic-xlarge-8p.json` 中（`basic-xlarge-8p` 的第 1000 回合），全部 130 只存活蚂蚁的 ID
# 均已超过 255，最大 ID 为 615。当某只蚂蚁阵亡时，其插槽永远不会被清空，因此一只出生 ID 为 300 的新蚂蚁会读到
# ID 为 44 的已阵亡蚂蚁遗留的数据。只有当一个座位的存活蚂蚁群跨越超过 `ANT_TABLE` 个连续 ID 时，两只存活的蚂蚁才会发生哈希碰撞 ——
# 在那一帧中并没有发生这种情况，因为同一批出生的蚂蚁具有近似连续的 ID —— 但继承一只已死蚂蚁的两个字节是普遍现象，而非例外。
#
# 这种机制虽然存在瑕疵，但它**完全具有自洽一致性**：适配器、NumPy 实现和计算图均以完全相同的方式进行回绕，
# 因此计算节点与训练器完全一致，`test_adapter_conformance` 依然成立。模型学到的是“这两个字节通常属于我，有时属于陌生人”，
# 并且它在天梯上学到的与训练中所见完全一致。加宽键位是治本之策但也意味着实质性代价：ID 需要在 u8 `ant_memory` 行中往返传递，
# 因此它需要增加第二个字节作为键，这需要全新的清单格式、新的计算图并对已发布的基线模型重新训练。已记录在 ants 的 *已知局限与缺口* 中。
#
# 与棋盘一样采用双重渲染：下方的适配器负责从视图构建计算图的三个输入，NumPy 函数为训练器构建相同的张量；
# `tests/test_adapter_conformance.py` 通过 `tinybrains adapt --obs` 在携带 `ant_memory` 的视图上运行适配器，
# 并确保两者严格相等。计算图的输入包括：
#
#   ant_planes  u8 [1, K, H, W]   每只蚂蚁记住的值写在它所在的单元格上
#   ids         i64 [1, N]        本回合模 ANT_TABLE 后的蚂蚁 ID，计算图会将其写回
#   cells       i64 [1, N]        每只蚂蚁所在单元格的扁平索引（row × cols + col），供计算图执行 gather

ANT_MEMORY = 2
ANT_TABLE = 256
ANT_DTYPE = "u8"
ANT_NP_DTYPE = np.uint8


def _ant_rows_logic() -> dict:
    """将上一回合的数据行重新索引匹配到本回合的蚂蚁：形状为 `[N, ANT_MEMORY]`（u8），即手册所述的多表连接（join）。"""
    # 形状为 `[1, P, K+1]` 的 `to_list` 是保存数据行的列表；reduce 负责将其投射提取出来，
    # 这是该语言用于索引程序计算值的方式。
    rows = {"reduce": [[{"to_list": [var("ant_memory")]}], var("current.0"), None]}
    cols = [
        {"scatter": [{"map": [rows, [var("0"), 0, var(str(k + 1))]]}, [ANT_TABLE, 1], ANT_DTYPE]}
        for k in range(ANT_MEMORY)
    ]
    keys = {"map": [var("ids"), {"%": [var(""), ANT_TABLE]}]}
    return {"gather": [{"concat": [cols, 1]}, keys, 0]}


def ant_planes_adapter(size) -> dict:
    """`ant_planes` 输入：每只蚂蚁重新对齐后的行数据散布在它所在的单元格上，在第 0 回合或没有蚂蚁时为全零。"""
    shape = {"merge": [[1, ANT_MEMORY], size]}
    joined = {"to_list": [{"concat": [[{"tensor": [var("mine"), ANT_DTYPE]}, _ant_rows_logic()], 1]}]}
    planes = [
        {"scatter": [{"map": [joined, [var("0"), var("1"), var(str(2 + k))]]}, size, ANT_DTYPE]}
        for k in range(ANT_MEMORY)
    ]
    return {
        "if": [
            {"and": [var("ant_memory"), var("mine")]},
            {"reshape": [{"stack": [planes, 0]}, shape]},
            {"zeros": [shape, ANT_DTYPE]},
        ]
    }


def ant_ids_adapter() -> dict:
    """`ids` 输入，形状为 `[1, N]`（i64）：本回合模哈希表后的蚂蚁 ID，计算图会原样写回。"""
    keys = {"map": [var("ids"), {"%": [var(""), ANT_TABLE]}]}
    return {"reshape": [{"tensor": [keys, "i64"]}, {"merge": [[1], [{"length": [var("ids")]}]]}]}


def ant_cells_adapter() -> dict:
    """`cells` 输入，形状为 `[1, N]`（i64）：每只蚂蚁在行优先展开下的单元格索引，供计算图的 gather 算子使用。
    在 map 内部当前上下文是单只蚂蚁，因此棋盘宽度需通过 `val` 从根节点读取。"""
    cell = {"+": [{"*": [var("0"), {"val": [[1], "size", 1]}]}, var("1")]}
    flat = {"map": [var("mine"), cell]}
    return {"reshape": [{"tensor": [flat, "i64"]}, {"merge": [[1], [{"length": [var("mine")]}]]}]}


def ant_rows_in_view(obs: dict) -> np.ndarray:
    """NumPy 实现的 `_ant_rows_logic`：`[N, ANT_MEMORY]`（u8），上一回合行按本回合 ids 读回。"""
    ids = obs.get("ids") or []
    out = np.zeros((len(ids), ANT_MEMORY), dtype=ANT_NP_DTYPE)
    prev = obs.get("ant_memory")
    if prev is None or not ids:
        return out
    table = np.zeros((ANT_TABLE, ANT_MEMORY), dtype=ANT_NP_DTYPE)
    for row in np.asarray(prev, dtype=np.int64).reshape(-1, ANT_MEMORY + 1):
        # 相同 ID 的后续数据行会覆盖先前的数据行，这与 `scatter` 的写入行为完全一致。
        table[int(row[0]) % ANT_TABLE] = np.clip(row[1:], 0, 255)
    return table[[int(i) % ANT_TABLE for i in ids]]


def ant_planes_in_view(obs: dict) -> np.ndarray:
    """NumPy 实现的 `ant_planes_adapter`：形状为 `[1, ANT_MEMORY, rows, cols]`（u8）。"""
    rows, cols = obs["size"]
    out = np.zeros((1, ANT_MEMORY, rows, cols), dtype=ANT_NP_DTYPE)
    if obs.get("ant_memory") is None or not obs["mine"]:
        return out
    values = ant_rows_in_view(obs)
    for (r, c), v in zip(obs["mine"], values):
        out[0, :, r, c] = v
    return out


def ant_ids_in_view(obs: dict) -> np.ndarray:
    return np.asarray([int(i) % ANT_TABLE for i in obs.get("ids") or []], dtype=np.int64).reshape(1, -1)


def ant_cells_in_view(obs: dict) -> np.ndarray:
    cols = obs["size"][1]
    return np.asarray([r * cols + c for r, c in obs["mine"]], dtype=np.int64).reshape(1, -1)
