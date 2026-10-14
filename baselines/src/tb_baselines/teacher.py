"""脚本启发式教师模型：全图势能场引导，每回合运行一次广度优先搜索（BFS）。

**它不是基准模型，绝不能作为参赛基准交付。** 它是一个*标注数据源（Label Source）*。规格等级天梯的全部意义在于：
固定同一个教师模型，并探究将其压缩至 8 KiB、64 KiB、512 KiB 时能保留多少能力 —— 这个问题唯有在每个等级的教师模型完全一致，
且该教师是一个规则智能体而非另一个神经网络时才具有科学意义。蒸馏启发式算法也是让天梯具备可对弈基线的最高效、诚实的方式；
未训练的静态桩代码原地不动，教不会任何人任何东西。

它是一个*对弈选手*，而不是第二套游戏引擎。它读取游戏协议定义的公开观测数据，并输出行动指令 —— 与选手的模型完全一致。
它不实现任何游戏规则：它从不裁决谁阵亡、蚁穴值多少分，或比赛何时结束。`ants/CLAUDE.md` 中的禁令禁止的是重复实现*游戏规则*，
而本模块没有任何规则逻辑。

## 它的对战机制

每个回合执行一次多源 BFS，在当前座位已知非水域的单元格上，同时从所有目标点出发扩散：食物、敌方蚁穴以及从未探索过的未知前沿。
每只蚂蚁随后沿着生成的距离势能场顺流而下。这是最经典的 Ants 开局智能体实现，且刻意保持平庸普通 ——
此处有趣的研究课题是神经网络能从中学到多少，而非该启发式算法本身有多强。

三项关键调优，每一项都旨在改变生成的对局质量而非改变智能体风格：

* **目标通过“起步领先（head start）”加权**，而非通过乘法系数加权。食物以距离 0 加入队列，敌方蚁穴以 `-hill_lead` 加入队列，
  因此蚁穴在相差该步数范围内具有更高优先级。在距离场上使用乘法系数会破坏 BFS 的单调性，那是此类实现的常见 Bug。
* **任意两只蚂蚁绝不抢占同一个方格**。否则它们都会阵亡（参见 `ants/engine/src/turn.rs`：停留在共享方格上的所有蚂蚁全部死亡，包括己方），
  那是损失整个蚁群最愚蠢的方式。
* **全程绝无任何随机性**。给定随机种子和游戏引擎，教师模型是关于观测数据的纯函数，这并非单纯的代码洁癖 ——
  这是使其完全可被模仿学习的前提保证。详见下文。

## 确定性是刚性要求，而非锦上添花

本模块的初始版本在决策前打乱了四个移动方向，以消除方向偏好并打破并列平局，且随机指派四分之一的蚂蚁进行盲目侦察。
这两项改动看似无害实则极具破坏性：在 116 万次决策实测中，标签分布呈现为 **W 25.6% / E 25.3% / S 24.3% / N 23.8% / 原地 1.0%** ——
几乎完全均匀分布，因为在开阔地形中，四个相邻单元格通常有两个与远方目标距离相同，此时的决策沦为抛硬币。

非状态函数的不可预测标签是无法被网络学习的。模仿学习的准确率存在一个上限，大致对应于非抛硬币决策的比例，再大的模型容量也无法突破该上限 ——
因此基于此类数据集构建的规格等级天梯，衡量的将是教师模型的噪声底噪，而非各等级的网络容量。

因此，平局按固定的方向顺序打破，且彻底废弃了随机侦察：探索前沿已经作为种子注入泛洪场中，因此当蚂蚁无处可去时，
它会走向未探索区域，*因为势能场正是这样指引的*。在真正平局时引入方向偏好是真实存在的，但这是完全正确的权衡 ——
它是确定性的，因此神经网络能够精准拟合它。

## 会记忆的教师（A teacher that remembers）

给定座位的记忆（`planes.MEMORY`，曾经出现过食物和敌方蚁穴的单元格，如 `collect.py` 在 `m` 键下携带的内容），
`orders` 会读取它，生成的训练标签也因此依赖于记忆：

* **见过一次的敌方蚁穴始终是目标，直到有蚂蚁走到该方格并确认其已被摧毁。** 普通教师模型在没有蚂蚁靠近蚁穴的回合会立即遗忘并四处游荡；
  而会记忆的教师会持续朝该目标行进。这是记忆在本作中的核心意义所在：摧毁蚁穴价值 2 分，且通常在离它很远时就已经被发现了。
* **视野之外曾经见过的食物是较弱的目标。** 食物在固定位置刷新（`ants/engine/src/food.rs` 循环对称集合），因此曾经出现过食物的单元格
  未来依然会刷新。它的种子优先级比视野内的食物低一级，比未知前沿高一级；若蚂蚁视野内能看到该单元格为空，则会跳过该点。

在不传入记忆时，它的动作与普通教师完全一致，步步吻合。两套逻辑合为一体，使得等级天梯保持其基准教师不变，
而记忆变体能够获得可通过记忆提升的更优标签：读取 `memory_in` 的模型有信号可学，不读取的模型则能够量化遗忘的代价。
"""

from __future__ import annotations

import numpy as np

from .planes import MOVES

# 行优先偏移量，顺序与 `MOVES` 完全一致，因此索引可直接对应。
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOLD = MOVES.index("-")

FAR = np.int32(1 << 29)


class Teacher:
    """每个进程维护一个实例即可；它在回合之间不保留任何状态，可安全复用。"""

    def __init__(self, hill_lead: int = 6, max_depth: int = 32):
        self.hill_lead = hill_lead
        # 蚂蚁每次仅决策一步，因此距离目标 60 格与距离目标 30 格会导出完全相同的移动决策。
        # 因此为泛洪深度设定上限在对局质量上是零损失的，同时占据了绝大部分运行时间优化：
        # 若不设上限，前沿种子会使得纯 Python 循环在每个座位的每个回合遍历棋盘的几乎每一个单元格。
        self.max_depth = max_depth

    # ---- 势能场构建 ----------------------------------------------------------------

    def field(self, obs: dict, water: np.ndarray, seen: np.ndarray,
              memory: dict | None = None) -> np.ndarray:
        """计算到最近目标的距离场，跨越已知可行走单元格并在环面上无缝环绕。

        未探索的单元格既是可行走的，*也是*目标点：这使得单个势能场能同时驱动采食与探索，
        也是视野内空无一物时蚁群依然向外扩散而非陷入原地发呆的原因。

        `memory` 是座位的记忆状态，格式为 `{plane: [[r, c], ...]}`，由 `planes.remember` 维护，
        若为遗忘型教师则传入 None（模块文档说明了其引入的差异）。

        **采用 NumPy 波前扩散，而非队列。** 在修复 `dilate` 后，被本实现替换的原版 Python BFS 占了训练循环耗时的 93% ——
        在两分钟的性能分析中累计进行了 1160 万次 deque 操作，因为前沿种子几乎将整个棋盘都变成了目标。
        一次性向外扩展整个圆环波前，用 `max_depth` 次数组运算完成了相同的泛洪扩散，取代了原先每个访问单元格一次 Python 迭代的开销。

        各层级通过偏移量平移，使得最早的种子从 0 开始：敌方蚁穴从 0 开始，食物从 `hill_lead` 开始，未知前沿比食物再晚两层。
        算法中仅比较差值，因此偏移量开销为零 —— 且起步领先是优先选择目标的正确方式，距离场上的乘法系数会破坏泛洪单调性，是常见的 Bug。
        """
        rows, cols = obs["size"]
        dist = np.full((rows, cols), FAR, dtype=np.int32)
        passable = ~water

        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int) -> None:
            if level in at_level:
                at_level[level] |= mask
            else:
                at_level[level] = mask.copy()

        hills = np.zeros((rows, cols), dtype=bool)
        for r, c, owner in obs["hills"]:
            if owner != 0:
                hills[r, c] = True
        if memory:
            # 座位本回合能够看到的视野：`seen` 是已知水域或当前可见区域，因此当前可见区域是其中非水域的部分。
            visible = seen & ~water
            # 记住的敌方蚁穴始终作为目标，除非其所在方格已进入视野且上面没有蚁穴（意味着已被摧毁）。
            # 记忆本身从不主动遗忘；教师模型对照当前视野对其进行校验，具备相同双输入的模型能够学会相同的判断逻辑。
            for r, c in memory.get("hill_foe_seen", ()):
                if not visible[r, c]:
                    hills[r, c] = True
        if hills.any():
            seed(hills, 0)

        if obs["food"]:
            food = np.zeros((rows, cols), dtype=bool)
            idx = np.asarray(obs["food"], dtype=np.int64)
            food[idx[:, 0], idx[:, 1]] = True
            seed(food, self.hill_lead)

        if memory:
            # 曾经见过但当前处于视野外的食物：食物会重新刷新的位置，优先级比视野内食物低一级。
            # 进入当前视野的记住单元格若无食物则直接跳过。
            recalled = np.zeros((rows, cols), dtype=bool)
            for r, c in memory.get("food_seen", ()):
                if not visible[r, c]:
                    recalled[r, c] = True
            if recalled.any():
                seed(recalled & passable, self.hill_lead + 1)

        unseen = ~seen
        if unseen.any():
            # 每一个未探索单元格，而非对其采样：在泛洪向量化后，没有任何理由对未知前沿进行近似，
            # 采样会导致蚁群对地图边界产生粗糙、不均的理解。
            seed(unseen & passable, self.hill_lead + 2)

        if not at_level:
            return dist

        wave = np.zeros((rows, cols), dtype=bool)
        for depth in range(0, self.max_depth + 1):
            step = at_level.pop(depth, None)
            if wave.any():
                spread = (
                    np.roll(wave, 1, 0) | np.roll(wave, -1, 0)
                    | np.roll(wave, 1, 1) | np.roll(wave, -1, 1)
                )
                step = spread if step is None else (spread | step)
            if step is None:
                continue
            step &= passable & (dist == FAR)
            if not step.any() and not at_level:
                break
            dist[step] = depth
            wave = step
        return dist

    # ---- 动作决策 ------------------------------------------------------------------

    def orders(self, obs: dict, water: np.ndarray, seen: np.ndarray,
               memory: dict | None = None) -> np.ndarray:
        """输出每只蚂蚁的移动动作索引，按位置与 `mine` 严格对齐。传入 `memory` 时，
        教师模型具备记忆能力（参见模块文档）；未传入时为遗忘型教师。"""
        mine = obs["mine"]
        if not mine:
            return np.zeros(0, dtype=np.int64)

        rows, cols = obs["size"]
        dist = self.field(obs, water, seen, memory)
        # 食物与水一样会阻挡移动（参见 `ants/engine/src/turn.rs`），因此指令移动到食物上的蚂蚁
        # 会留在原地 —— 这白白浪费了回合，更糟的是会误导神经网络以为不会被阻挡。
        blocked = np.zeros((rows, cols), dtype=bool)
        for r, c in obs["food"]:
            blocked[r, c] = True

        taken = {tuple(a) for a in mine}
        out = np.full(len(mine), HOLD, dtype=np.int64)

        # 距离最近的蚂蚁优先决策，确保对方格具有最强占有权的蚂蚁获得该方格。
        # 排序平局按蚂蚁索引打破，对应 `mine` 的顺序（引擎固定为行优先）—— 因此这也是完全确定性的。
        order = sorted(range(len(mine)), key=lambda i: (dist[mine[i][0], mine[i][1]], i))
        for i in order:
            ar, ac = mine[i]
            best, best_d = HOLD, dist[ar, ac]
            # 固定方向顺序，且仅在严格更优时才转移。在开阔地带两个等距邻居非常普遍，
            # 若随机选择会导致四分之一的标签无法被网络学习。
            for k in range(4):
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                if water[nr, nc] or blocked[nr, nc] or (nr, nc) in taken:
                    continue
                if dist[nr, nc] < best_d:
                    best, best_d = k, dist[nr, nc]
            if best != HOLD:
                dr, dc = DELTAS[best]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                taken.discard((ar, ac))
                taken.add((nr, nc))
            out[i] = best
        return out


def known(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """从单次观测中提取已知水域以及曾经探索过的全部区域。

    `water` 是唯一自带记忆的字段，因此 1 表示见过的已知水域，0 表示*已知陆地*或者*从未探索过的单元格*。
    教师模型需要区分这两者，从单次观测中提取这一信息的严谨方法是计算己方蚂蚁周围的可见圆盘 ——
    这与第 6 个特征平面计算的内容相同，也是其存在的同一原因。
    """
    from .planes import VIEW_RADIUS2, Board

    rows, cols = obs["size"]
    b = Board(rows, cols)
    water = b.rle(obs["water"]["rle"]).astype(bool)
    # 单元格若为已知水域或当前处于可见视野内，则视为“已探索”。这会略有漏算 —— 很久以前走过但现在处于迷雾中的陆地
    # 会被读为未探索 —— 但在此处这是完全良性的偏差：它激励蚁群持续向外探索，而不是过早认定地图已探索完毕。
    visible = b.dilate(b.scatter(obs["mine"]), VIEW_RADIUS2).astype(bool)
    return water, water | visible
