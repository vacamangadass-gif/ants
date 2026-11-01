"""`manifest.json`，由 `planes.py` 生成 —— 绝不可手工编写。

无人手工编辑的构件绝不会与其生成来源产生偏离。这与 `ants/build.sh` 应用于 `cartridge.json`
和 `plugin.json` 的规则相同，且在此处适用该规则有着更为尖锐的原因：适配器是特征编码的两套渲染实现之一，
如果有人能够修改其中一套而不同步另一套，整个防偏差（anti-skew）逻辑就会彻底崩溃。

    python -m tb_baselines.adapters > manifest.json

其字节序列是平台计算哈希的输入，也是其衡量大小的一半（`S' = artifact_bytes + len(manifest)`），
因此这必须是**确定性的**：紧凑的分隔符、排序的键、无末尾换行符。改变空格格式就会导致 `manifest_hash` 改变，
对于含义完全相同的文档而言，这将产生一个全新的提交。

## 清单（Manifest）的构成

Orion 的 `orion:model@1.0.0`：模型名称、带有数据类型和形状的输入输出，以及每个输入对应的一个 JSONLogic **适配器（adapter）**，
该适配器负责将游戏观测数据转换为该输入对应的张量。这是参赛选手编写的全部接口表面 —— 绝无第二份文档。

这里只有一个针对 `board` 输入的适配器，没有输出解码器。平台直接读取网络头部：清单的 `result` 表达式仅针对输出张量求值，
因此它无法直接看到观测数据，也无法在蚂蚁所在的单元格执行聚集操作。`kalam-match` 负责执行聚集操作，且本清单声明了它将找到的头部形状 —— 每个单元格 `[1, moves, H, W]`。

当 `memory=True` 时，清单会声明第二个输入 `memory_in`，其适配器负责传递上一回合的记忆（在第 0 回合为全零），
以及第二个输出 `memory`，运行器在下一回合的视野中将其原样传回（参见 `planes.MEMORY` 及手册中的 *Memory* 章节）。

## 为什么形状轴使用命名变量

`H` 和 `W` 是可变维度轴（Orion 1.8.1）。一个赛季会运行多种尺寸的棋盘，轴名称会与调用传入的实际尺寸绑定，
因此单份清单和单个加载的模型会话即可服务从 24x24 到 120x124 的所有棋盘。`probe_dims` 是准入系统执行 5 次零填充推理时所使用的尺寸，
128x128 覆盖了任何棋盘可能具有的最长边（124，对应 cartridge.json 中的 `limits.boards`）：若探测最小尺寸，则会放行没人游玩的超小棋盘。
"""

from __future__ import annotations

import json
import sys

from .planes import (ANT_DTYPE, ANT_MEMORY, DTYPE, N_MEMORY, N_MOVES, N_PLANES, PLANES,
                     ant_cells_adapter, ant_ids_adapter, ant_planes_adapter, memory_adapter)

ABI = "orion:model@1.0.0"

# 准入探测将每个命名轴绑定到的具体尺寸。至少为任何棋盘可能具有的最长边
# （124，对应 `limits.boards`），因为 `probe_ms` 的代表性取决于其运行的尺寸。
PROBE_DIMS = {"H": 128, "W": 128}


def board_adapter() -> dict:
    """将观测数据转换为 `board` 张量，形状为 `[1, planes, H, W]`。"""
    size = {"var": "size"}
    stacked = {"stack": [[p.logic(size) for p in PLANES], 0]}
    # `merge` 在运行时构建形状列表：[1, planes] ++ size。`reshape` 仅为元数据，
    # 开销为 1，因此开头的 batch 轴是零成本的。
    return {"reshape": [stacked, {"merge": [[1, N_PLANES], size]}]}


def manifest(name: str = "tb.baseline", version: str = "1", memory: bool | str = False,
             ants: bool | str = False) -> dict:
    """`memory` 为 False、True 或 `"where"`（默认）、`"max"`（`planes.MEMORY`，`i8`）或 `"learned"`（相同的端口）；
    `ants` 为 False、True 或 `"learned"`（每只蚂蚁对应 `planes.ANT_MEMORY` 个数值）。"""
    mkind = ("where" if memory is True else memory) if memory else None
    akind = ("learned" if ants is True else ants) if ants else None
    inputs = [
        {
            "name": "board",
            "dtype": DTYPE,
            "shape": [1, N_PLANES, "H", "W"],
            "adapter": board_adapter(),
        }
    ]
    # 平台从中聚集策略输出的网络头部。采用 `f32` 是因为策略输出为得分而非离散类别，
    # 且通道顺序遵循游戏的动作定义（`planes.MOVES`）。
    outputs = [{"name": "policy", "dtype": "f32", "shape": [1, N_MOVES, "H", "W"]}]
    description = "A TinyBrains Ants entry: seven planes in, a per-cell policy out."
    if mkind in ("where", "max", "learned"):
        # 记忆与棋盘一样采用 `i8`：两个 0/1 平面，按每个单元格 2 字节计费。
        shape = [1, N_MEMORY, "H", "W"]
        inputs.append({"name": "memory_in", "dtype": DTYPE, "shape": shape,
                       "adapter": memory_adapter({"var": "size"})})
        outputs.append({"name": "memory", "dtype": DTYPE, "shape": shape})
        description = ("A TinyBrains Ants entry: seven planes and a two-plane memory in, "
                       "a per-cell policy and the memory out.")
    elif mkind:
        raise ValueError(f"no memory '{mkind}'")
    if akind:
        # 单蚁记忆：计算图在蚂蚁所在单元格读取其记住的值，并输出每只蚂蚁一行（其 ID 在前），
        # 运行器负责在下一回合将其传回。
        size = {"var": "size"}
        k = ANT_MEMORY
        inputs += [
            {"name": "ant_planes", "dtype": ANT_DTYPE, "shape": [1, k, "H", "W"], "adapter": ant_planes_adapter(size)},
            {"name": "ids", "dtype": "i64", "shape": [1, "N"], "adapter": ant_ids_adapter()},
            {"name": "cells", "dtype": "i64", "shape": [1, "N"], "adapter": ant_cells_adapter()},
        ]
        outputs.append({"name": "ant_memory", "dtype": ANT_DTYPE, "shape": [1, "N", k + 1]})
        description += " Each ant carries a row of its own."
    return {
        "abi": ABI,
        "name": name,
        "version": version,
        "format": "onnx",
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
        # 单蚁记忆命名了一个蚂蚁轴，探测必须将其绑定到某个具体数值。
        "probe_dims": PROBE_DIMS | ({"N": 8} if akind else {}),
    }


def dumps(name: str = "tb.baseline", version: str = "1", memory: bool | str = False,
          ants: bool | str = False) -> str:
    """输出精确字节。经排序并保持紧凑，确保相同的规范始终产生相同的哈希值。"""
    return json.dumps(manifest(name, version, memory, ants), separators=(",", ":"), sort_keys=True)


def main() -> None:
    """`python -m tb_baselines.adapters [NAME] [--memory[=KIND]] [--ants[=KIND]]`"""
    memory: bool | str = False
    ants: bool | str = False
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--memory"):
            memory = a.split("=", 1)[1] if "=" in a else True
        elif a.startswith("--ants"):
            ants = a.split("=", 1)[1] if "=" in a else True
        else:
            args.append(a)
    sys.stdout.write(dumps(args[0] if args else "tb.baseline", memory=memory, ants=ants))


if __name__ == "__main__":
    main()
