"""将 PyTorch 模型导出为可提交的参赛构件：ONNX 格式、fp16 权重初始化器，以及通过平台官方裁决验证。

最后一步并非本仓库自行发明的检查逻辑。它通过调用外部子进程 `tinybrains check --json` 执行，
该工具完全按照平台准入系统的方式读取计算图，并在计算节点所运行的相同评估器上执行清单 ——
因此“是否契合其声明的规格等级”是基于平台在卡带自身参考观测数据上进行的完全相同的测量标准来判定的。
本模块在此基础上增加了完整的闭环控制：构建模型、测量大小与耗时，并直接拒绝未能契合其规格等级的违规构件。

    python -m tb_baselines.export --class micro --weights runs/micro-bc/best.pt --out models/micro-bc

每次导出都会在模型文件旁生成三个配套文件：`manifest.json`（自动生成，严禁手工编辑）、
`metrics.json`（平台裁决测量的指标数据）以及 `card.md`（人工复现所需的完整说明卡片）。
模型卡片注明了**引擎摘要（engine digest）**，因为无法说明针对哪个引擎版本训练的基准模型是任何人都无法无损复现的。
（评估器采用 datalogic，`tinybrains games` 会打印出该二进制文件链接的评估器版本。）
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from . import adapters, nets
from .planes import ANT_DTYPE, ANT_MEMORY, MEMORY, N_MEMORY, N_PLANES

ROOT = Path(__file__).resolve().parents[2]
OPSET = 17  # 位于部署环境支持的 13-19 范围之内，且与基准测试 fixture 一致


# ---- 规格等级表读取 ----------------------------------------------------------------------

def classes() -> dict:
    with open(ROOT / "classes.toml", "rb") as f:
        doc = tomllib.load(f)
    return {c["name"]: c for c in doc["class"]} | {"_": doc}


def budget(name: str) -> dict:
    """计算单个规格等级允许的各项预算，直接由 `classes.toml` 推导计算而非重复记录。"""
    table = classes()
    spec = table[name]
    doc = table["_"]
    per_param = doc["metric"][f"{doc['dtype']['default']}_bytes_per_param"]
    # 某个等级可自定义其目标填充率：`small` 无法在回合截止时限内达到全局 70% 的填充目标，
    # 若强行统一会导致所有 small 构件因非自身问题而认证失败。
    fraction = spec.get("fraction", doc["target"]["fraction"])
    target = int(spec["max_bytes"] * fraction)
    weights = target - doc["metric"]["manifest_bytes"]
    c = doc["compute"]
    return {
        "name": name,
        "max_bytes": spec["max_bytes"],
        "fraction": fraction,
        "target_bytes": target,
        "weight_bytes": weights,
        "max_share": doc["target"]["max_share"],
        "share_us": c["turn_ms"] * 1000 / c["rows_per_call"],
        "params_at_target": int(weights / per_param),
        "flops_per_seat_turn": c["turn_ms"] / 1000 / c["rows_per_call"] * c["flops_per_second"],
        "spec": spec,
    }


# ---- PyTorch 导出为 ONNX ----------------------------------------------------------------

def to_onnx(trunk: torch.nn.Module, path: Path, planes: int = N_PLANES) -> None:
    """仅导出策略网络部分。

    H 和 W 设为动态轴，因为一个赛季包含多种棋盘尺寸，且清单将它们声明为命名轴 `"H"` 和 `"W"` ——
    单个准入通过的推理会话即可服务赛季中游玩的所有棋盘。开头的 batch 轴同样为动态轴，
    尽管清单将其固定为 1：一场比赛为两个座位，每个座位为独立的 `model_infer` 调用，因此线上不执行批处理，
    但将该轴在计算图中保持符号化无需额外成本，且能使构件兼容支持批处理的训练器。
    """
    trunk.eval()
    example = (torch.zeros(1, planes, 128, 128, dtype=torch.int8),)
    inputs, outputs = ["board"], ["policy"]
    axes = {"board": {0: "B", 2: "H", 3: "W"}, "policy": {0: "B", 2: "H", 3: "W"}}
    # 记忆模型在两个方向上各有两个端口，命名与生成的清单声明严格一致。记忆与棋盘一样采用 `i8`。
    if getattr(trunk, "memory_ports", False):
        example += (torch.zeros(1, N_MEMORY, 128, 128, dtype=torch.int8),)
        inputs, outputs = inputs + ["memory_in"], outputs + ["memory"]
        axes |= {"memory_in": {0: "B", 2: "H", 3: "W"}, "memory": {0: "B", 2: "H", 3: "W"}}
    # 单蚁记忆会增加适配器的三个输入和单蚁输出行；蚂蚁轴设为动态轴，因为蚁群规模随时在变。
    if getattr(trunk, "ant_ports", False):
        n = 8
        k = ANT_MEMORY
        example += (torch.zeros(1, k, 128, 128, dtype=torch.uint8),
                    torch.arange(n, dtype=torch.int64).reshape(1, n),
                    torch.arange(n, dtype=torch.int64).reshape(1, n) * 129)
        inputs, outputs = inputs + ["ant_planes", "ids", "cells"], outputs + ["ant_memory"]
        axes |= {"ant_planes": {0: "B", 2: "H", 3: "W"}, "ids": {0: "B", 1: "N"},
                 "cells": {0: "B", 1: "N"}, "ant_memory": {0: "B", 1: "N"}}
    # TorchScript 导出器已被弃用并推荐使用 dynamo，但 dynamo 目前尚无法生成本图在所有棋盘尺寸上所需的动态 H/W 轴。
    # 此处刻意固定抑制该警告；待 dynamo 完善支持后再行调整。
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")
    torch.onnx.export(
        trunk, example, str(path),
        input_names=inputs, output_names=outputs,
        dynamic_axes=axes,
        opset_version=OPSET, dynamo=False,
    )


def halve(path: Path) -> None:
    """将每个 float32 初始化器重写为 float16，并在其使用处通过 `Cast` 转回。

    由于大小指标直接衡量构件的原始字节数，这直接将规格等级衡量的权重体积减半 ——
    且比过去更直接（过去指标首先会对初始化器数据执行 zstd 压缩）。计算图的运行时精度不受影响：
    运行时在优化计算图时会对 `Cast(initializer)` 执行常量折叠，因此 fp32 张量在会话加载时重建一次，
    绝不在每次推理时重复转换。

    实测数据：在相同规格等级下参数量达 2.03 倍，推理时间无感知变化，算子依然完全处于白名单内，
    且在 3 场实战对弈中 396 次单蚁决策与 fp32 计算图 100% 逐一完全一致。
    """
    model = onnx.load(str(path))
    g = model.graph
    casts, kept = [], []
    for init in g.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype != "float32":
            kept.append(init)
            continue
        kept.append(numpy_helper.from_array(array.astype("float16"), init.name + "_h"))
        casts.append(
            onnx.helper.make_node(
                "Cast", [init.name + "_h"], [init.name],
                to=onnx.TensorProto.FLOAT, name="cast_" + init.name,
            )
        )
    if not casts:
        return
    del g.initializer[:]
    g.initializer.extend(kept)
    # ONNX 要求拓扑顺序，Cast 必须位于其使用者之前，因此将它们置于最前。
    rest = list(g.node)
    del g.node[:]
    g.node.extend(casts + rest)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def shorten(path: Path) -> None:
    """剔除所有节点名称，并将所有中间张量重命名为短 ID。

    节点和张量名称纯粹占用大小指标的字节数，而运行时根本不读取它们：导出器会为原本只需 `t7` 的张量写入
    `/trunk/stage/convs.0/Conv_output_0`，且节点名称完全是可选的。计算图的输入输出保持原名，
    因为清单直接声明了它们；初始化器保持自身名称，因为 `halve` 增加的 `Cast` 依赖它读取。
    在拥有几百个节点的计算图中，这能节省数十千字节，在 nano 等级中这几乎占了整个等级的大部分预算。
    """
    model = onnx.load(str(path))
    g = model.graph
    fixed = {i.name for i in g.input} | {o.name for o in g.output} | {t.name for t in g.initializer}
    names: dict[str, str] = {}

    def short(name: str) -> str:
        if not name or name in fixed:
            return name
        if name not in names:
            names[name] = f"t{len(names)}"
        return names[name]

    # 每个不同数值共享单个 `Constant` 节点：导出器会为每个切片的 starts, ends, axes, steps
    # 创建独立的常量节点，导致几百个节点的图携带数百份 `[0]`、`[1]` 和 `[-1]` 的冗余副本。
    seen: dict[bytes, str] = {}
    alias: dict[str, str] = {}
    kept = []
    for node in g.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            key = node.attribute[0].SerializeToString()
            if key in seen:
                alias[node.output[0]] = seen[key]
                continue
            seen[key] = node.output[0]
        kept.append(node)
    del g.node[:]
    g.node.extend(kept)
    for node in g.node:
        node.name = ""
        for i, name in enumerate(node.input):
            node.input[i] = short(alias.get(name, name))
        for i, name in enumerate(node.output):
            node.output[i] = short(name)
    for vi in list(g.value_info):
        vi.name = short(vi.name)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# ---- 平台官方裁决 -----------------------------------------------------------------------

def cli() -> str:
    found = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")
    if not found:
        raise SystemExit(
            "PATH 中未找到 `tinybrains` 二进制文件。\n"
            "  brew tap tiny-brains/cli https://github.com/Tiny-Brains/cli\n"
            "  brew install tiny-brains/cli/tinybrains   (或将 TINYBRAINS 设置为该可执行文件路径)"
        )
    return found


def verdict(model: Path, manifest: Path) -> dict:
    """`tinybrains check --json`：节点自身的真实评估器与运行时，在参考测试集上运行。"""
    r = subprocess.run(
        [cli(), "check", str(model), str(manifest), "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if not r.stdout.strip():
        raise SystemExit(f"tinybrains check 未产生任何输出：\n{r.stderr}")
    return json.loads(r.stdout)


def certify(name: str, said: dict) -> list[str]:
    """汇总该构件存在的全部合规问题，按严重性依次排列。"""
    b = budget(name)
    size = said["size_metric_bytes"]
    problems = []

    if not said["ok"]:
        problems.append(
            f"平台拒绝了该构件: {said.get('reason') or '未指明原因'}".strip()
        )
    if size > b["max_bytes"]:
        problems.append(
            f"S' 大小为 {size:,} 字节，而 {name} 等级上限为 {b['max_bytes']:,} —— 超出 "
            f"{size - b['max_bytes']:,} 字节"
        )
    # 回合截止时限检查。超过 `mini` 等级后这才是拒绝构件的主因，字节上限根本无从发力。
    #
    # 一个座位的时限份额为整个回合时间：每个座位执行一次 `model_infer`，各自拥有独立的超时限制，
    # 因此不存在需要拆分的共享调用。`budget()` 保留了本仓库为自身设定的安全边界，
    # 这是自我约束的缓冲空间，而非平台强制的硬性规则。
    share = said["infer_us_max"] / b["share_us"]
    if share > b["max_share"]:
        problems.append(
            f"在最差参考棋盘上消耗了单座 {b['share_us'] / 1000:.1f} ms 时限份额的 {share:.0%}，"
            f"超出了本仓库允许的 {b['max_share']:.0%} 上限。该等级尚有字节余量，但回合时间已耗尽。"
        )
    # 记忆费用从清单计算并与具体赛季的等级配额对比，本仓库并不预知赛季配置，
    # 因此 `check` 仅报告费用而不做通过裁决。但它确实会检验往返有效性：当喂入自身输出的记忆时，计算图必须能成功响应。
    mem = said.get("memory")
    if mem:
        if mem.get("verdict"):
            problems.append(f"记忆配置被拒绝: {mem['verdict']} -- {mem.get('reason')}")
        rt = mem.get("round_trip") or {}
        if rt.get("failed"):
            problems.append(
                f"在喂入自身记忆时，计算图在 {rt['checked']} 次链式调用中有 {rt['failed']} 次失败 "
                f"(测试用例 {rt.get('failing_case')}: {rt.get('reason')})"
            )
    return problems


def report(name: str, said: dict) -> dict:
    b = budget(name)
    g = said["graph"]
    return {
        "class": name,
        # S' = artifact_bytes + len(manifest)。两项均为计算节点在重新哈希后测得的真实大小，
        # 因此绝不会因权重存放位置的不同而产生瞒报 —— 原先基于初始化器 zstd 压缩的旧指标就存在此类缺陷。
        "size_metric_bytes": said["size_metric_bytes"],
        "class_max_bytes": b["max_bytes"],
        "class_target_bytes": b["target_bytes"],
        "fill_of_cap": round(said["size_metric_bytes"] / b["max_bytes"], 4),
        "artifact_bytes": said["artifact_bytes"],
        "manifest_bytes": said["manifest_bytes"],
        "params": g["parameters"],
        "nodes": g["nodes"],
        "opset": g["opset"],
        "operators": g["operators"],
        "adapter_ops_max": said["ops_max"],
        "adapter_ops_budget": said["budget_ops"],
        "infer_us_max": said["infer_us_max"],
        "deadline_share_ms": round(b["share_us"] / 1000, 2),
        "share_used": round(said["infer_us_max"] / b["share_us"], 4),
        "target_fraction": b["fraction"],
        "engine_digest": said["engine_digest"],
        "weights_hash": said["weights_hash"],
        "manifest_hash": said["manifest_hash"],
        # 当模型声明记忆时，仅根据清单计算出的记忆费用：固定字节、每单元格字节，
        # 以及在规格包络中最小和最大棋盘上的总体费用。
        "memory": said.get("memory"),
    }


CARD = """# {name}

{summary}

| | |
|---|---|
| 重量规格等级 (Weight class) | **{cls}** — {size:,} / {cap:,} 字节（占用上限的 {fill:.0%}） |
| 参数量 (Parameters) | {params:,} ({dtype} 初始化器) |
| 网络架构 (Architecture) | `{arch}`，每个方向感受野 **{reach} 格**{dil} |
| 训练方法 (Method) | {method} |
| 适配器开销 (Adapter) | 在最差参考用例下消耗 {ops:,} / {opsmax:,} 算子操作 |
| 推理耗时 (Inference) | 在最差参考用例下耗时 {infer:.2f} ms — 占单个座位 {share:.0f} ms 时限的 **{shareuse:.1%}** |
| 算子清单 (Operators) | {operators} |
| 测量基准引擎 (Measured on) | engine `{engine}` |
| 模型哈希 (Model hash) | `{whash}` |
| 清单哈希 (Manifest hash) | `{ahash}` |
{memory}
{notes}

复现命令 (Reproduce with):

```sh
{repro}
```

推理时间是在运行检查的具体机器上测量的，**仅供展示，绝非准入门禁**：
平台没有设置算力上限。此处列出耗时是因为过于庞大的计算图会触碰回合截止时间限制，
而单个座位的时限份额为整个回合时间 —— 每个座位执行一次独立的 `model_infer` 调用，各自拥有独立的超时限制。

**Measured on** 是上述各项指标测量时所依据的游戏引擎摘要，而非强制要求完全一致的锁定项。
starter 运行时采用其 `games.toml` 中固定的 `engine`，该版本会随着 ants 的每次发版而更新；
上述指标在当前指定的引擎上真实有效，且不会随着每个新版本的发布而重新测量，因为引擎的变动并不改变已导出的模型本身。
若需获取最新环境下的实时数值，可自行运行 `tinybrains check` 进行重测。
"""


def memory_row(metrics: dict) -> str:
    """提取模型跨回合携带的状态信息，渲染为模型卡片的 Memory 行：包括棋盘记忆、单蚁记忆或两者兼备，
    以及 `check` 从清单中读取的综合费用报告。"""
    mem = metrics.get("memory")
    if not mem:
        return ""
    kind, ants = metrics.get("memory_kind"), metrics.get("ant_memory")
    learned = {"learned": "计算图自学习"}
    carries = []
    if kind or not ants:
        what = learned.get(kind, ", ".join(m.name for m in MEMORY))
        carries.append(f"`{N_MEMORY}` 个 `i8` 特征平面 ({what})")
    if ants:
        carries.append(f"每只蚂蚁对应一行包含 `{ANT_MEMORY}` 个 `{ANT_DTYPE}` 数值并按 ID 追踪 "
                       f"({learned.get(ants, ants)})")
    # 当 `check` 无法计算记忆费用时（如不支持的形状 `MEMORY_SHAPE` 或发布未声明 `limits.boards`），
    # 依然会返回对象结构，仅在数值处填充 null（参见 `cli/src/cmd/check.rs` 中的 `MemoryReport::json`）。
    # 模型卡片忠实陈述已知信息并明确指明缺失项：`certify` 负责将拒绝判定转化为终止导出的错误信息，
    # 而 `write_card` 先行执行，因此避免格式化崩溃掩盖真正的错误。
    cell, most = mem.get("cell_bytes"), mem.get("bytes_at_max")
    rt = mem.get("round_trip") or {}
    priced = (f"每个单元格 {cell} 字节: 在最大棋盘上共计 {most:,} 字节"
              if cell is not None and most is not None else "`check` 无法计算其费用")
    judged = (f"`check` 在 {rt['checked']} 次链式观测中评测了往返往复"
              if rt.get("checked") is not None else "`check` 未执行往返评测")
    return f"| 记忆机制 (Memory) | {' 以及 '.join(carries)}，{priced}。赛季等级必须允许该费用；{judged} |\n"


def write_card(out: Path, name: str, method: str, metrics: dict, summary: str,
               notes: str, repro: str) -> None:
    (out / "card.md").write_text(CARD.format(
        memory=memory_row(metrics),
        name=name, summary=summary, cls=metrics["class"],
        size=metrics["size_metric_bytes"], cap=metrics["class_max_bytes"],
        fill=metrics["fill_of_cap"], params=metrics["params"],
        dtype=classes()["_"]["dtype"]["default"], method=method,
        arch=metrics.get("arch", "?"), reach=metrics.get("reach_cells", 0),
        dil=(f" (空洞率序列 {metrics['dilations'][0]})"
             if metrics.get("dilations") else ""),
        ops=metrics["adapter_ops_max"], opsmax=metrics["adapter_ops_budget"],
        infer=metrics["infer_us_max"] / 1000, share=metrics["deadline_share_ms"],
        shareuse=metrics["share_used"],
        operators=", ".join(metrics["operators"]), engine=metrics["engine_digest"],
        whash=metrics["weights_hash"], ahash=metrics["manifest_hash"],
        notes=notes, repro=repro,
    ))


def shape_of(trunk: torch.nn.Module) -> dict:
    """提取网络架构属性，记录在模型卡片和 metrics 中。

    **过时的构件目录具有欺骗性的自洽性**：其模型、适配器和 metrics 完全互相吻合，
    各项哈希检查也全部通过，因此没有任何机制能发现它是由本仓库已不存在的旧版本代码生成的。
    此前几乎就发布了这样一个错误构件 —— 感受野修复之前的旧 `nano-bc` 紧挨着修复之后的新 `micro-bc`。
    将具体网络架构白纸黑字记录下来，使得此类问题能一目了然地被发现，而无需通过反推参数量来排查。
    """
    stages = [m for m in trunk.modules() if type(m).__name__ == "Stage"]
    inner = getattr(trunk, "trunk", trunk)
    kind = getattr(trunk, "memory_kind", None)
    ants = getattr(trunk, "ant_kind", None) if getattr(trunk, "ant_ports", False) else None
    return {
        "arch": type(inner).__name__ + (f" + {kind} memory" if kind else "")
                + ({"learned": " + a memory per ant"}.get(ants, "")),
        "memory_kind": kind,
        "ant_memory": ants,
        "reach_cells": max((st.reach for st in stages), default=0),
        "dilations": [st.dilations for st in stages] or None,
    }


def command(module: str, argv: list[str]) -> str:
    """格式化可以在本目录下直接输入的完整命令行字符串。"""
    return shlex.join(["python", "-m", module, *argv])


def recipe(weights: Path | None, argv: list[str]) -> str:
    """提取生成构件的完整命令链，每行一条，填充进模型卡片的 Reproduce 模块中。

    每个步骤都会自行记录其命令行（`collect` 写入数据集的 `.stats.json`，训练器写入训练运行的 `history.json`），
    因此此处直接读取它们，而非盲目猜测运行中可能覆盖的参数标志：数据集采集命令、训练命令以及本次导出命令。
    在支持该功能前的旧版本运行记录不会提供命令，此时模型卡片将仅显示本次导出命令。
    """
    steps = []
    history = weights.parent / "history.json" if weights else None
    if history and history.exists():
        h = json.loads(history.read_text())
        if h.get("data"):
            stats = Path(h["data"]).with_suffix("").with_suffix(".stats.json")
            if stats.exists() and json.loads(stats.read_text()).get("command"):
                steps.append(json.loads(stats.read_text())["command"])
        if h.get("command"):
            steps.append(h["command"])
    steps.append(command("tb_baselines.export", argv))
    return "\n".join(steps)


def export(trunk: torch.nn.Module, name: str, out: Path, method: str, repro: str,
           summary: str = "", notes: str = "") -> dict:
    """执行全部导出流水线。若构件未能契合其规格等级则抛出异常中止。

    `repro` 为模型卡片中的 Reproduce 模块：记录重新生成该构件的完整命令序列。
    """
    out.mkdir(parents=True, exist_ok=True)
    model_path, manifest_path = out / "model.onnx", out / "manifest.json"
    manifest_path.write_text(adapters.dumps(
        f"tb.{out.name}",
        memory=getattr(trunk, "memory_kind", None) or getattr(trunk, "memory_ports", False),
        ants=getattr(trunk, "ant_kind", None) or getattr(trunk, "ant_ports", False)))
    to_onnx(trunk, model_path)
    if classes()["_"]["dtype"]["default"] == "fp16":
        halve(model_path)
    shorten(model_path)

    said = verdict(model_path, manifest_path)
    problems = certify(name, said)
    metrics = report(name, said) | shape_of(trunk)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_card(out, out.name, method, metrics,
               summary or f"A {name}-class Ants policy.", notes, repro)

    if problems:
        raise SystemExit(
            f"{out}: 该构件未能契合规格等级 {name}\n  "
            + "\n  ".join(problems)
        )
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    # 变体模型通过命令行参数控制，而非维护第二张表格：运行生成的 history.json 和模型卡片
    # 均会记录实际构建的网络结构，那才是真正关键的记录依据。
    ap.add_argument("--arch", default=None, help="覆盖规格等级默认的网络架构")
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--weights", type=Path, help=".pt 格式的权重字典；缺省时采用随机初始化")
    ap.add_argument("--memory", nargs="?", const="max", default=False,
                    help="构建携带记忆的模型：`max`（planes.MEMORY）或 `learned`；"
                         "当权重中声明记忆时会自动从中读取")
    ap.add_argument("--ants", nargs="?", const="learned", default=False,
                    help="构建同时携带单蚁记忆的模型：`learned`；同样可从权重中自动读取")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="untrained")
    ap.add_argument("--repro", default=None,
                    help="模型卡片的 Reproduce 复现模块内容（默认：数据集采集、训练运行及本次导出的自动记录命令）")
    argv = sys.argv[1:] if argv is None else argv
    a = ap.parse_args(argv)

    from .train.bc import override
    spec = override(classes()[a.cls], a)
    state = torch.load(a.weights, map_location="cpu") if a.weights else {}
    memory = a.memory or state.get("memory", False)
    ants = a.ants or state.get("ants", False)
    trunk = nets.build(spec, memory=memory, ants=ants)
    if a.weights:
        trunk.load_state_dict(state.get("trunk", state))

    b = budget(a.cls)
    print(f"{a.cls}: {nets.policy_params(trunk):,} 个参数，"
          f"预算目标约为 {b['params_at_target']:,} 参数（在 {b['target_bytes']:,} 字节目标下）")
    m = export(trunk, a.cls, a.out, a.method, a.repro or recipe(a.weights, argv))
    print(f"  S = {m['size_metric_bytes']:,} 字节，占 {a.cls} 上限的 {m['fill_of_cap']:.0%} "
          f"（设定目标为 {m['target_fraction']:.0%}）")
    print(f"  适配器算子数 {m['adapter_ops_max']:,} ops，推理耗时 {m['infer_us_max'] / 1000:.2f} ms "
          f"= 占单座时限份额的 {m['share_used']:.0%}")
    if m.get("memory"):
        mem = m["memory"]
        rt = mem.get("round_trip") or {}
        # 当 `check` 无法计算费用时填充 null；参见 `memory_row`。
        if mem.get("cell_bytes") is not None and mem.get("bytes_at_max") is not None:
            print(f"  记忆费用：每个单元格 {mem['cell_bytes']} 字节，在最大棋盘上共计 {mem['bytes_at_max']:,} 字节")
        else:
            print("  已声明记忆，但 `check` 无法计算其具体费用")
        if rt.get("checked") is not None:
            print(f"  往返测试：在 {rt['checked']} 次链式调用中有 {rt['failed']} 次失败")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
