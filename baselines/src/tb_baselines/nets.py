"""神经网络架构定义，每个规格等级对应一套网络结构。

塑造此处所有网络架构的核心要素有三点，且没有一条是“通常在图像处理中有效”的常规套路：

**棋盘具有环面环绕性。** Ants 棋盘是环面网格，因此采用零填充（zero-padded）的卷积会向模型传递一种错误信号：
即在网格边缘存在一堵墙，而实际上边缘的另一侧正是地图的反方向。此处的每个卷积层均通过拼接边缘切片实现环绕填充（wrapping），
其导出为 `Slice` + `Concat` 算子 —— 两者均在平台算子允许白名单中，而 `padding_mode="circular"` 会导出为仅从 opset 18
才开始支持的 `Pad` 模式。

**棋盘包含多种尺寸。** 边长从 24 到 124 不等（对应卡带的 `limits.boards`），且一个赛季的棋盘在赛季进行中会持续变化。
全卷积架构（Fully Convolutional）天然无缝支持任意尺寸，这也是始终坚持全卷积架构的主要原因。当网络进行下采样时（`EncDec`），
步长必须能整除其运行的每个棋盘边长：基础棋盘的边长全部能被 4 整除，但赛季 1 的许多棋盘不能（如 26、45、49、105 等），
因此带步长的网络在适于提交前需要填充到步长倍数。

**超过 `mini` 等级后，回合时间限制在达到字节上限前就先生效。** 参见 `classes.toml` 中的 `[compute]` 部分。
因此天梯并不是简单地将同一种架构按比例放大：`trunk` 在全分辨率下消耗其参数预算，而 `encdec` 将大部分参数转移到步长下采样阶段，
在该阶段单个参数的计算成本仅为全分辨率下的四分之一或十六分之一。

## 感受野而非参数量是真正的刚性约束

此处最早训练的模型是 `Trunk(channels=48, blocks=1)` —— 一个 3x3 stem 浅层，一个 3x3 block 残差块，随后接 1x1 卷积。
这构成了 **5x5 的感受野**：一只蚂蚁每个方向只能感知 2 格距离，而它所模仿的教师模型是在深度达 32 格的泛洪范围内进行规划。
nano 和 micro 等级均在 1 个 epoch 内于 44-47% 吻合度处陷入停滞，实战表现远逊于教师模型，在教师模型能发展到 33 只蚂蚁时，
它们只能分别发展出 5 只和 15 只蚂蚁的群落 —— 且 micro 增加的参数量几乎没有带来任何提升，因为瓶颈根本不是模型容量。

空洞卷积（Dilation）以零额外开销解决了这一问题。`Conv` 将空洞率作为固有属性，无需增加任何白名单外的新算子，
且在每层将空洞率翻倍仅需 4 层即可达到 31x31 的感受野，而无空洞的普通堆叠则需要 16 层。

价值头（Value Head）**仅用于训练且绝不导出**。这是一项切实的资源节约 —— 在 nano 等级中它会占去三分之一的预算 ——
同时也是放置特权信息的合理位置：评判网络（Critic）可以看到真实分数（`tinybrains env --scores every`），
因为评判网络在任何正式对战之前就已经被彻底丢弃。策略网络仅接收经过战争迷雾过滤后的视野，除此无它。

## Padding 填充在每个阶段只执行一次，而非逐层执行

本文件的初始版本在每个 3x3 卷积前都执行环绕填充，其实测速度**比 FLOP 计算量预测的慢了 2.3 倍** ——
三个卷积意味着六次全张量 concat 拼接。无填充的 `n` 层 3x3 卷积堆叠会使棋盘缩小 `2n`，因此在最前端预先填充 `n` 个单元格，
在末端恰好保留原始尺寸，仅需一对 concat 拼接，在 128x128 尺寸下仅多产生约 10% 的内部计算量。
完全相同的数学运算，完全相同的结果，而张量复制开销降低到三分之一。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .planes import ANT_MEMORY, MEMORY_SOURCES, N_MEMORY, N_MOVES, N_PLANES, VIS_SOURCE


def wrap(x: torch.Tensor, k: int) -> torch.Tensor:
    """在全部四个方向上对 `k` 个单元格进行环面环绕填充，通过拼接边缘切片实现。

    不采用 `F.pad(mode="circular")`：因为该算子会导出为带 `wrap` 的 ONNX `Pad`，需要 opset 18 支持，
    且部署环境的允许白名单是按具体算子而非按模式检查的。Slice 与 Concat 语义明确，且均在白名单中。
    """
    if k == 0:
        return x
    x = torch.cat([x[..., -k:], x, x[..., :k]], dim=-1)
    x = torch.cat([x[..., -k:, :], x, x[..., :k, :]], dim=-2)
    return x


class Stage(nn.Module):
    """一组无填充的 3x3 卷积层，仅在最前端执行一次统一环绕填充。

    棋盘在输出时与输入尺寸完全一致，因为空洞率为 `d` 的 3x3 卷积会使棋盘缩小 `2d`，
    而前端的 wrap 在每一侧增加了 `sum(dilations)` 个单元格。

    **空洞卷积是本类存在的根本原因。** 参见模块文档字符串：普通 3x3 卷积堆叠每层仅增加 2 格感受野，
    在此训练的初始模型仅能看到 2 格距离且实战表现极差。每层将空洞率翻倍可实现指数级感受野增长 ——
    1, 2, 4, 8 仅用 4 层即可达到 31x31 感受野 —— 且除更宽的环绕填充外无需任何额外计算开销。
    ONNX 将空洞率作为 `Conv` 的属性而非独立算子承载，因此无需白名单外的任何算子：
    空洞图经静态检查仅包含 `Cast, Concat, Constant, Conv, Relu, Slice`。
    """

    def __init__(self, widths: list[tuple[int, int]], dilations: list[int] | None = None):
        super().__init__()
        self.dilations = dilations or [1] * len(widths)
        assert len(self.dilations) == len(widths)
        self.convs = nn.ModuleList([
            nn.Conv2d(a, b, 3, padding=0, dilation=d)
            for (a, b), d in zip(widths, self.dilations)
        ])

    @property
    def reach(self) -> int:
        """输出单元格所能感知的空间距离（格数）。该数值决定了策略是能追踪行进痕迹，还是仅能感知自身所在的孤立方格。"""
        return sum(self.dilations)

    def forward(self, x):
        x = wrap(x, self.reach)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i + 1 < len(self.convs):
                x = F.relu(x)
        return F.relu(x)


def doubling(n: int, cap: int = 8) -> list[int]:
    """生成 1, 2, 4, 8, 8, ... —— 针对 `n` 层的空洞率序列，在达到 `cap` 上限后停止翻倍。

    超过 8 之后，采样空洞比食物分布密度还要宽，卷积核会开始对噪声采样，因此后续层保持固定不再翻倍。
    4 层即可在每个方向触达 15 格距离，大致相当于视野半径（半径平方 77，约 8.8 格）加上规划移动路径的必要余量。
    """
    return [min(cap, 1 << i) for i in range(n)]


class Trunk(nn.Module):
    """全分辨率下的全卷积架构：对应 nano 和 micro 等级。

    包含一个空洞 3x3 卷积 stage，随后接若干 1x1 卷积层；1x1 卷积仅以 3x3 九分之一的开销增加网络深度 ——
    在总预算仅有几千个参数时这是绝佳的折衷设计。不使用归一化层：在这些通道宽度下归一化参数占了预算的可观比例，
    且输入特征平面本身已是 0/1 二值平面。

    `dilate=False` 用于复现早期模型使用的无空洞堆叠架构。保留它是为了对比其实验发现，而非仅作为技术备注。
    """

    def __init__(self, channels: int, blocks: int, planes: int = N_PLANES, moves: int = N_MOVES,
                 dilate: bool = True):
        super().__init__()
        widths = [(planes, channels)] + [(channels, channels)] * blocks
        dilations = doubling(len(widths)) if dilate else None
        self.stage = Stage(widths, dilations)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.head = nn.Conv2d(channels, moves, 1)
        self.channels = channels

    def features(self, x):
        return F.relu(self.mix(self.stage(x)))

    def forward(self, board):
        return self.head(self.features(board.float()))


class EncDec(nn.Module):
    """在中间层执行步长下采样并在该阶段分配参数预算，最后上采样恢复分辨率：对应 mini 和 small 等级。

    较窄的全分辨率 stem 保留单格细节并保持低成本开销；主体在下采样步长 `stride` 下运行，
    在该尺度下单个参数的计算成本仅为全分辨率下的 1/stride²；随后通过最近邻上采样与 stem 进行跳跃拼接，
    使 1x1 头部依然能够看到它所做决策的具体网格单元。

    采用 `Resize` 作为上采样算子是因为 **`ConvTranspose` 不在允许的算子白名单中**，
    采用 `avg_pool` 作为下采样算子是出于同样考量：步长为 2 的卷积虽然也行，但会占用本规格等级更希望投入到深度的参数量。
    """

    def __init__(
        self, channels: int, stride: int, blocks: int,
        planes: int = N_PLANES, moves: int = N_MOVES,
    ):
        super().__init__()
        assert stride in (2, 4), "下采样步长必须能整除模型运行的任意棋盘边长"
        self.stride = stride
        stem_ch = max(8, channels // 4)
        # stem 保持无空洞状态：它在全分辨率下运行，职责是捕捉单格细节。
        # 感受野扩展交由深度堆叠完成，在下采样阶段单个单元格已经代表棋盘的 stride 格范围。
        self.stem = Stage([(planes, stem_ch)])
        deep_widths = [(stem_ch, channels)] + [(channels, channels)] * blocks
        self.deep = Stage(deep_widths, doubling(len(deep_widths), cap=4))
        self.up = nn.Conv2d(channels, stem_ch, 1)
        self.channels = stem_ch * 2
        self.head = nn.Conv2d(self.channels, moves, 1)

    def features(self, x):
        skip = self.stem(x)
        y = self.deep(F.avg_pool2d(skip, self.stride))
        y = F.relu(self.up(y))
        y = F.interpolate(y, scale_factor=self.stride, mode="nearest")
        return torch.cat([skip, y], dim=1)

    def forward(self, board):
        return self.head(self.features(board.float()))


class PerCell(nn.Module):
    """仅包含 1x1 卷积：具备相同容量但**完全没有空间感受野**。

    这是方法列的对照组。它仅能感知蚂蚁所处单元格的 7 个特征数值 —— 此处是否有水、是否有蚁穴、是否有己方蚂蚁 ——
    而对其周围相邻单元格一无所知，因此它无法追踪食物踪迹、无法躲避战斗，也无法向未知前沿推进。
    赋予它与 `Trunk` 相同的规格等级、相同的数据集和大致相同的参数量，两者的差距恰恰量化了空间感受野的真正价值，
    这也是在本作中严谨证明空间上下文价值的唯一科学方法。

    逐单元格模型也是整个项目的下限基准线。如果卷积主干网络无法击败它，说明该主干并没有学会观察四周环境。
    """

    def __init__(self, channels: int, blocks: int = 1, planes: int = N_PLANES,
                 moves: int = N_MOVES):
        super().__init__()
        widths = [(planes, channels)] + [(channels, channels)] * blocks
        self.layers = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in widths])
        self.head = nn.Conv2d(channels, moves, 1)
        self.channels = channels

    def features(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

    def forward(self, board):
        return self.head(self.features(board.float()))


class WithMemory(nn.Module):
    """读取棋盘及座位记忆，并将更新后的记忆写回的主干网络包装器。

    记忆结构对应 `planes.MEMORY`：视图在回合间遗忘的两个 0/1 平面。
    更新模式支持 'where'（默认，基于视野掩码的条件选择更新，清除已摧毁敌丘幽灵残留）
    与 'max'（传统并集累积，供消融实验使用）。
    它是座位历史所见内容的固定函数，因此除读取它的策略网络外无任何可学习参数。主干网络基于
    `N_PLANES + N_MEMORY` 个输入通道构建，接收棋盘与记忆堆叠后的张量。更新计算在 float 下运行并强转回原类型，
    输出 `i8`，与清单声明及棋盘一致的数据类型。
    `export.to_onnx` 将端口命名为 `board`, `memory_in`, `policy` 和 `memory`，与生成的清单完全匹配。
    """

    memory_ports = True     # 导出时在 `board` 和 `policy` 旁附带 `memory_in` 与 `memory`
    memory_kind = "where"

    def __init__(self, trunk: nn.Module, mode: str = "where"):
        super().__init__()
        self.trunk = trunk
        self.channels = trunk.channels
        self.mode = mode
        self.memory_kind = mode
        self.register_buffer("sources", torch.tensor(MEMORY_SOURCES, dtype=torch.long))

    def read(self, memory_in: torch.Tensor) -> torch.Tensor:
        """主干网络所看到的记忆平面。"""
        return memory_in.float()

    def write(self, f: torch.Tensor, board: torch.Tensor, memory_in: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """写入下一回合的记忆。"""
        m = self.mode if mode is None else mode
        seen = board.index_select(1, self.sources).float()
        if m == "where":
            vis = (board[:, 6:7] != 0)
            return torch.where(vis, seen, memory_in.float()).to(torch.int8)
        elif m == "max":
            return torch.maximum(memory_in.float(), seen).to(torch.int8)
        else:
            raise ValueError(f"未知的记忆更新模式 '{m}'（可用项：'where', 'max'）")

    def forward(self, board: torch.Tensor, memory_in: torch.Tensor):
        f = self.trunk.features(torch.cat([board.float(), self.read(memory_in)], dim=1))
        return self.trunk.head(f), self.write(f, board, memory_in)


class LearnedMemory(nn.Module):
    """记忆由计算图自主学习的主干网络：每回合根据座位所见和所记内容写入 `N_MEMORY` 个平面，
    与 `WithMemory` 一样以 `i8` 格式传递。

    不同于 `WithMemory` 使用 `Max` 保持两个固定平面，本类由优化器塑造两个特征平面 ——
    相当于一个卷积 GRU，其输入变换即为主干网络本身。`memory_in` 以 [-127, 127] 的 `i8` 传入，
    读取为 `h = memory_in / 127`；主干网络接收堆叠的棋盘与 `h`；基于特征与 `h` 的 1x1 门控和候选卷积输出
    `h' = (1 - z) h + z tanh(c)`；输出为强转为 `i8` 的 `Floor(h' x 127 + 0.5)`，由运行器传递给下一回合。
    `Round` 不在算子白名单中，而 `Floor`, `Cast`, `Sigmoid` 和 `Tanh` 在。

    **训练过程中同样传递舍入后的值。** 前向传播执行舍入，反向传播采用恒等映射（直通估计器 STE），
    因此网络在下一回合学会读取的内容与运行器传回的内容完全吻合，避免学习线缆无法传输的冗余精度。
    清单格式与 `WithMemory` 相同：两个 `i8` 平面，每个单元格 2 字节，因此允许前者的规格等级同样允许本架构；
    使用 `train/seq.py` 作为训练器，因为计算图自主计算的记忆必须按照座位的时序进行回放训练。
    """

    memory_ports = True
    memory_kind = "learned"

    def __init__(self, trunk: nn.Module, planes: int = N_MEMORY):
        super().__init__()
        self.trunk = trunk
        self.channels = trunk.channels
        self.planes = planes
        self.gate = nn.Conv2d(trunk.channels + planes, planes, 1)
        self.cand = nn.Conv2d(trunk.channels + planes, planes, 1)

    def read(self, memory_in: torch.Tensor) -> torch.Tensor:
        return memory_in.float() / 127.0

    def write(self, f: torch.Tensor, board: torch.Tensor, memory_in: torch.Tensor) -> torch.Tensor:
        h = self.read(memory_in)
        fh = torch.cat([f, h], dim=1)
        z = torch.sigmoid(self.gate(fh))
        c = torch.tanh(self.cand(fh))
        scaled = ((1 - z) * h + z * c) * 127.0
        rounded = torch.floor(scaled + 0.5)
        if self.training:
            # 向前传递的值是舍入后的值；反向传播梯度则取自未舍入的值。
            return scaled + (rounded - scaled).detach()
        return rounded.to(torch.int8)

    def forward(self, board: torch.Tensor, memory_in: torch.Tensor):
        f = self.trunk.features(torch.cat([board.float(), self.read(memory_in)], dim=1))
        return self.trunk.head(f), self.write(f, board, memory_in)


class PerAnt(nn.Module):
    """在棋盘记忆旁附带单蚁记忆，或独立存在：每只蚂蚁终生保留 `planes.ANT_MEMORY` 个数值，
    如 `LearnedMemory` 一样学习并在 `ant_memory` 下公开传递。

    输入对应适配器的三个接口（参见 `planes.py` 中的*单蚁记忆*）：`ant_planes`（每只蚂蚁记住的值写在自身单元格上，类型为 `u8`）、
    `ids`（本回合模哈希表后的蚂蚁 ID）与 `cells`（每只蚂蚁所在单元格的行优先索引）。
    主干网络接收棋盘、棋盘记忆（若存在）以及读取为 `u8 / 127.5 - 1` 的单蚁特征平面；基于特征的 1x1 门控和候选卷积
    为每个单元格赋予新值 `(1 - z) a + z tanh(c)`，该值仅在有蚂蚁停留的位置具有实际意义。
    输出端通过 `GatherElements`（在白名单中）将这些单元格聚集为 `[1, N, K]`，通过 `Floor` 和 `Cast` 舍入回 `u8`，
    并在每行最前端写入对应蚂蚁的 ID：每只蚂蚁一行，运行器在下一回合原样传回，适配器按 ID 重新匹配，
    确保蚂蚁在任意移动中状态保持跟随，新蚂蚁读取到初始全零。

    `inner` 可为 `WithMemory`, `LearnedMemory` 或 None；当指定内部记忆时，计算图同时拥有两套记忆，
    清单同时声明两个输出端口并合并计费。
    """

    ant_ports = True

    def __init__(self, trunk: nn.Module, inner: nn.Module | None = None, planes: int = ANT_MEMORY,
                 kind: str = "learned"):
        super().__init__()
        self.trunk = trunk
        self.inner = inner
        self.channels = trunk.channels
        self.kind = kind
        self.planes = planes
        self.gate = nn.Conv2d(trunk.channels + planes, planes, 1)
        self.cand = nn.Conv2d(trunk.channels + planes, planes, 1)

    ant_kind = property(lambda self: self.kind)

    @property
    def memory_ports(self) -> bool:
        return self.inner is not None

    @property
    def memory_kind(self) -> str | None:
        return getattr(self.inner, "memory_kind", None)

    def forward(self, board: torch.Tensor, *rest: torch.Tensor):
        if self.inner is not None:
            memory_in, ant_planes, ids, cells = rest
        else:
            memory_in, (ant_planes, ids, cells) = None, rest
        a = ant_planes.float() / 127.5 - 1.0
        parts = [board.float()]
        if self.inner is not None:
            parts.append(self.inner.read(memory_in))
        parts.append(a)
        f = self.trunk.features(torch.cat(parts, dim=1))
        outputs = [self.trunk.head(f)]
        if self.inner is not None:
            outputs.append(self.inner.write(f, board, memory_in))

        fa = torch.cat([f, a], dim=1)
        z = torch.sigmoid(self.gate(fa))
        c = torch.tanh(self.cand(fa))
        scaled = ((1 - z) * a + z * c + 1.0) * 127.5          # [B, K, H, W] 范围在 [0, 255]
        rounded = torch.floor(scaled + 0.5)
        value = scaled + (rounded - scaled).detach() if self.training else rounded
        flat = value.flatten(2)                                 # [B, K, H*W]
        index = cells.unsqueeze(1).expand(-1, self.planes, -1)  # [B, K, N]
        rows = torch.gather(flat, 2, index).transpose(1, 2)     # [B, N, K]
        head = ids.unsqueeze(-1).to(rows.dtype)                 # [B, N, 1]
        ant_memory = torch.cat([head, rows], dim=2)
        outputs.append(ant_memory if self.training else ant_memory.to(torch.uint8))
        return tuple(outputs)


ARCHS = {"trunk": Trunk, "encdec": EncDec, "percell": PerCell}

MEMORIES = {"where": WithMemory, "max": WithMemory, "learned": LearnedMemory}
MEMORY_PLANES = {"where": N_MEMORY, "max": N_MEMORY, "learned": N_MEMORY}
ANT_PLANES = {"learned": ANT_MEMORY}


def build(spec: dict, memory: bool | str = False, ants: bool | str = False) -> nn.Module:
    """根据 `classes.toml` 中的规格配置构建 PyTorch 模型；指定 `memory` 时构建携带棋盘记忆的模型：
    `"where"`（或 True）代表基于可见性掩码 Where 维护的 `planes.MEMORY`；
    `"max"` 代表通过 `Max` 维护的传统记忆（供消融使用）；
    `"learned"` 代表 `LearnedMemory`；
    指定 `ants` 时同时携带单蚁记忆：`"learned"`（或 True）由计算图自主学习。"""
    arch = spec["arch"]
    if arch not in ARCHS:
        raise ValueError(f"未知的网络架构 '{arch}'（可用项：{', '.join(sorted(ARCHS))}）")
    kwargs = {k: spec[k] for k in ("channels", "stride", "blocks", "dilate") if k in spec}
    mkind = ("where" if memory is True else memory) if memory else None
    akind = ("learned" if ants is True else ants) if ants else None
    if mkind and mkind not in MEMORIES:
        raise ValueError(f"未知的记忆类型 '{mkind}'（可用项：{', '.join(sorted(MEMORIES))}）")
    if akind and akind not in ANT_PLANES:
        raise ValueError(f"未知的单蚁记忆类型 '{akind}'（可用项：{', '.join(sorted(ANT_PLANES))}）")
    if mkind or akind:
        kwargs["planes"] = N_PLANES + (MEMORY_PLANES[mkind] if mkind else 0) + (ANT_PLANES[akind] if akind else 0)
    net = ARCHS[arch](**kwargs)
    if mkind in ("where", "max"):
        inner = WithMemory(net, mode=mkind)
    elif mkind:
        inner = MEMORIES[mkind](net)
    else:
        inner = None
    if akind:
        return PerAnt(net, inner, kind=akind)
    return inner or net


class ActorCritic(nn.Module):
    """优化器所见的主干网络与价值头。`policy` 最终发布上线；`value` 在导出时被丢弃。

    评判网络（Critic）读取与策略网络完全相同的特征，并可选择读取环境报告的特权数值 ——
    这对于非对称 Actor-Critic 是标准做法，在此处极其安全且有必要明确说明原因：
    该执行路径上的任何内容都无法存活通过 `export.py`，因此任何特权数值都绝不可能进入实际比赛的模型中。
    """

    def __init__(self, trunk: nn.Module, privileged: int = 0):
        super().__init__()
        self.trunk = trunk
        self.privileged = privileged
        self.value = nn.Sequential(
            nn.Linear(trunk.channels + privileged, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, board: torch.Tensor, extra: torch.Tensor | None = None):
        f = self.trunk.features(board.float())
        logits = self.trunk.head(f)
        # 每个座位汇聚为一个向量。棋盘全局平均池化是与尺寸无关的最廉价池化方式，且价值头不需要空间位置细节。
        pooled = f.mean(dim=(2, 3))
        if self.privileged:
            assert extra is not None, "该评判网络设计用于接收特权输入"
            pooled = torch.cat([pooled, extra], dim=1)
        return logits, self.value(pooled).squeeze(-1)


def policy_params(model: nn.Module) -> int:
    """规格等级衡量所针对的参数量：仅统计最终导出的策略部分。记忆包装器完整保留发布（门控位于计算图中）；
    Actor-Critic 仅导出策略主干并剔除价值头。"""
    # 判定“该模块是否整体交付”，而非“是否携带棋盘记忆”：当 PerAnt 未包裹内部记忆时（`--memory none --ants`），
    # 其 `memory_ports` 为 False，但其门控和候选层依然会被完整导出，因此必须以整体交付逻辑进行参数统计。
    whole = getattr(model, "memory_ports", False) or getattr(model, "ant_ports", False)
    exported = model if whole else getattr(model, "trunk", model)
    return sum(p.numel() for p in exported.parameters())
