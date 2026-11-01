"""训练器的编码器与天梯的适配器必须生成完全相同的张量。逐字节一致。

**这是本仓库中最重要的测试。** 这里的其他代码都是训练脚本，其错误可以通过指标异常察觉。
而这个测试防范的故障则毫无征兆：一旦编码器与适配器产生分歧，模型所依据的训练分布
将永远不会在天梯竞技场中出现，唯一的表象就是天梯分低于训练曲线的预期。

这不是自我验证的重新实现。`tinybrains adapt` 通过 **datalogic** —— 即 Orion 节点编译适配器所用的求值器，
且采用与集群完全一致的版本 —— 在卡带自带的参考观测数据上评估清单，
而这正是对齐一致性具有决定意义的唯一基准数据集。若本测试通过，
则证明在平台实际验证的输入集上，`planes.encode` 与 `manifest.json` 具有完全一致的函数行为。

    pytest tests/ -q                    # 需要 PATH 中存在 `tinybrains`，或设置 TINYBRAINS 环境变量

记忆机制（`planes.MEMORY`）同样遵循这套严苛标准，分三部分检验：通过 `tinybrains adapt --obs`
（支持记忆传递的 CLI）对比 `memory_in` 适配器与 `planes.memory_in_view`；
对比 numpy 版 `remember` 与计算图所用的 `Max` 算子；以及在安装了 torch 的环境下验证计算图自身。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines import adapters, planes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")


@pytest.fixture(scope="session")
def dumped(tmp_path_factory):
    """天梯的官方求值结果：清单的适配器通过 datalogic —— Orion 节点运行它们所用的求值器 ——
    在卡带的参考观测数据集上的执行产物。

    `adapt` 也会加载计算图，因此如果清单声明的形状被计算图拒绝，会在本测试直接报错而非拖到准入阶段。
    任何已训练的模型均可作为载体：编码逻辑归清单所有，而非权重。
    本仓库不提交任何预训练权重 —— 平台的基准模型存放在 ants-starter 中 ——
    因此计算图优先读取环境变量 `$TB_CONFORMANCE_ONNX`，否则取本检出目录旁的 starter 的 micro-bc 模型。
    若显式指定了环境变量但文件不存在，则判为 FAILURE 而非跳过：CI 会设置该变量，而在 CI 中跳过的门禁形同虚设。"""
    if not CLI:
        pytest.skip("no `tinybrains` on PATH; set TINYBRAINS to the binary")
    named = os.environ.get("TB_CONFORMANCE_ONNX")
    onnx = Path(named) if named else ROOT.parents[1] / "ants-starter" / "models" / "micro-bc" / "model.onnx"
    if not onnx.exists():
        if named:
            pytest.fail(f"TB_CONFORMANCE_ONNX names {onnx}, which does not exist")
        pytest.skip(f"no trained graph to load the manifest against: set TB_CONFORMANCE_ONNX, or check out ants-starter beside ants ({onnx})")
    dist = ROOT.parent / "dist"
    if not dist.exists():
        pytest.skip(f"no prebuilt dist/ at {dist}; run ./build.sh or provide release artifacts")
    out = tmp_path_factory.mktemp("tensors")
    manifest = out / "manifest.json"
    manifest.write_text(adapters.dumps())
    r = subprocess.run(
        [CLI, "adapt", str(onnx), str(manifest), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"tinybrains adapt failed:\n{r.stdout}\n{r.stderr}"
    return json.loads((out / "index.json").read_text()), out


def test_every_reference_observation_encodes_identically(dumped):
    index, out = dumped
    assert index["cases"], "the cartridge shipped no reference observations"

    for case in index["cases"]:
        i = case["case"]
        obs = json.loads((out / f"case-{i}" / "observation.json").read_text())
        theirs = np.load(out / f"case-{i}" / "board.npy")
        ours = planes.encode(obs)

        assert ours.shape == theirs.shape, (
            f"case {i}: the trainer builds {ours.shape}, the adapter builds {theirs.shape}"
        )
        assert ours.dtype == theirs.dtype, f"case {i}: {ours.dtype} vs {theirs.dtype}"

        if not np.array_equal(ours, theirs):
            # 报告具体的特征面而非孤立单元格。通常是整个特征面出现偏差，
            # 报告“平面 4 (hill_mine) 存在 2 处单元格不一致”远比报告一维扁平索引更直观有效。
            bad = [
                f"{k} ({planes.PLANES[k].name}): {int((ours[0, k] != theirs[0, k]).sum())} cells"
                for k in range(planes.N_PLANES)
                if not np.array_equal(ours[0, k], theirs[0, k])
            ]
            pytest.fail(f"case {i} disagrees on plane(s) -- " + "; ".join(bad))


def test_the_adapter_fits_the_operation_budget(dumped):
    index, _ = dumped
    budget = index["budget_ops"]
    worst = max(c["ops_in"] for c in index["cases"])
    # 准入校验拒绝的是最差情况，因此平均值不是关注的指标。需要保持充裕的安全裕度，
    # 而不仅仅是刚过及格线：因为实际对局中可能出现比参考数据集更密集的极端观测。
    assert worst < budget * 0.6, (
        f"the adapter costs {worst} of {budget} at its worst -- too little headroom for a "
        f"board busier than the reference set"
    )


def test_the_generated_manifest_is_byte_stable():
    """平台会对这些字节进行哈希 —— 并计算其大小，因为 `S'` 为
    `artifact_bytes + len(manifest)` —— 因此相同的规范必须始终产生逐字节一致的文档。"""
    assert adapters.dumps() == adapters.dumps()
    doc = json.loads(adapters.dumps())
    assert doc["abi"] == adapters.ABI
    assert [i["name"] for i in doc["inputs"]] == ["board"], "the graph declares one input"
    assert [o["name"] for o in doc["outputs"]] == ["policy"]
    assert "result" not in doc, "the platform reads the head; a manifest may not decode it"
    # 维度轴带有显式命名，这使得单个 session 能够服务赛季运行的任何棋盘。
    assert doc["inputs"][0]["shape"][2:] == ["H", "W"]
    assert doc["outputs"][0]["shape"][2:] == ["H", "W"]
    assert doc["probe_dims"] == adapters.PROBE_DIMS


def test_the_encoder_handles_an_empty_colony():
    """存活蚂蚁为零始终是合法状态（参见《模型所见》文档），且全灭的座位直到比赛结束前都会持续发送该状态。
    索引空列表的编码器会在此处崩溃。"""
    obs = {
        "size": [64, 96],
        "mine": [], "foes": [], "food": [], "hills": [],
        "water": {"rle": [0, 64 * 96]},
        # 无蚂蚁即无视野：引擎发送全零掩码，这与 `ants` 在
        # `the_view_carries_the_mask_it_filtered_through` 中断言的行为一致。
        "vis": {"rle": [0, 64 * 96]},
    }
    board = planes.encode(obs)
    assert board.shape == (1, planes.N_PLANES, 64, 96)
    assert board.sum() == 0, "nothing seen, nothing set -- including the visibility mask"


def test_the_action_table_is_the_channel_order_the_platform_decodes():
    """策略头的第 `i` 个通道代表 `MOVES[i]`。

    该索引对应交叉熵标签所使用的映射（`train/bc.py` 中的 `MOVE_INDEX`），
    也是 `orders_from_indices` 写回动作的依据。**闭环由平台完成，而非清单**：
    `kalam-match` 对各通道执行 argmax 并在其自带的动作表中索引，
    该表在 `kalam/scripts/gen-kalam.py` 和 `cli/src/model.rs` 中硬编码为 `["N","E","S","W","-"]`。
    因此该表是训练器与平台之间的契约，中间没有任何适配器介入 —— 若两者顺序不一致，
    模型在对局中的每个动作都将发生系统性错乱，而模型推理、损失计算、回放和对局却全都能正常运行，
    其他任何环节都不会察觉该错误。
    """
    assert list(planes.MOVES) == ["N", "E", "S", "W", "-"], (
        "the platform's decode table is fixed in kalam and the CLI; a reorder here is a silent "
        "relabelling of every move the ladder plays"
    )
    assert planes.N_MOVES == json.loads(adapters.dumps())["outputs"][0]["shape"][1], (
        "the manifest declares a head whose channel count is not the move count"
    )


def test_the_manifest_reads_the_output_the_export_names():
    """`export.to_onnx` 将计算图的输出命名为 `policy`，清单也声明同名输出。
    单方面重命名会导致准入阶段因找不到该张量而被拒绝，这是明确的致命错误 ——
    但这要求有人显式运行准入检查，而此测试更轻量。"""
    doc = json.loads(adapters.dumps())
    assert [o["name"] for o in doc["outputs"]] == ["policy"]


def test_the_visibility_plane_reads_the_engines_mask():
    """`vis` 是由引擎直接发送的，而非派生计算得出。

    旧版特征面曾使用 `tb.dilate(scatter(mine), 77)`，其存在是因为模型若缺少它
    便无法区分“已知为空”与“从未探索”。表达式语言无法引用外层迭代器的元素，
    因此单蚁视野圆盘只能硬编码为展开 241 次的卷积核，或者完全不支持 ——
    而视野半径属于卡带自有的游戏规则。本测试断言编码器直接读取引擎发送的掩码而非自行重建。"""
    doc = json.loads(adapters.dumps())
    text = json.dumps(doc)
    assert "vis.rle" in text, "the visibility plane must read the observation's `vis`"
    assert "dilate" not in text, "the dilate operator is withdrawn; the engine sends the mask"


def test_the_engines_mask_is_the_disk_it_claims_to_be(dumped):
    """独立的第三方校验，也是信任引擎下发掩码安全可靠的原因。

    `Board.dilate` 正是为此保留：它像训练器以往那样从 `mine` 计算圆盘并集，
    引擎发送的 `vis` 在每个参考观测上必须与之严格相等。
    若卡带未来修改了视野规则，测试将在此处直接失败，而非体现在天梯分的异常上。"""
    index, out = dumped
    for case in index["cases"]:
        i = case["case"]
        obs = json.loads((out / f"case-{i}" / "observation.json").read_text())
        rows, cols = obs["size"]
        b = planes.Board(rows, cols)
        sent = b.rle(obs["vis"]["rle"])
        derived = b.dilate(b.scatter(obs["mine"]), planes.VIEW_RADIUS2)
        assert np.array_equal(sent, derived), (
            f"case {i}: the engine's `vis` is not the radius-{planes.VIEW_RADIUS2} disk union of "
            f"`mine` -- {int((sent != derived).sum())} cells differ"
        )


# ---- 记忆机制 -------------------------------------------------------------------------

def cases_of(dumped) -> list[dict]:
    index, out = dumped
    return [json.loads((out / f"case-{c['case']}" / "observation.json").read_text())
            for c in index["cases"]]


def chained(observations: list[dict], per_size: int = 12):
    """`(前序状态, 观测)` 组合对，按棋盘尺寸串联，模拟比赛中的时序链路：
    记忆依附于棋盘，因此单个座位的状态绝不会跨越不同棋盘尺寸。"""
    by_size: dict[tuple[int, int], list[dict]] = {}
    for o in observations:
        by_size.setdefault(tuple(o["size"]), []).append(o)
    for group in by_size.values():
        state = None
        for o in group[:per_size]:
            yield state, o
            state = planes.remember(state, o)


def test_the_memory_manifest_declares_the_two_ports_beside_the_board():
    """`memory_in` 为第二个输入端口，`memory` 为第二个输出端口，均为 `[1, N_MEMORY, H, W]` 尺寸的 `i8`；
    纯棋盘清单不受此选项影响。"""
    doc = json.loads(adapters.dumps(memory=True))
    assert [i["name"] for i in doc["inputs"]] == ["board", "memory_in"]
    assert [o["name"] for o in doc["outputs"]] == ["policy", "memory"]
    for port in (doc["inputs"][1], doc["outputs"][1]):
        assert port["dtype"] == planes.DTYPE
        assert port["shape"] == [1, planes.N_MEMORY, "H", "W"]
    text = json.dumps(doc["inputs"][1]["adapter"])
    assert '{"tensor": [{"var": "memory"}]}' in text, "the memory passes through as the tensor it is"
    assert '"zeros"' in text, "turn 0 has no memory, so the adapter builds zeros"
    assert adapters.dumps() == adapters.dumps(memory=False)
    assert json.loads(adapters.dumps())["inputs"][0] == doc["inputs"][0], "the board adapter is the same"


def test_remember_is_the_where_the_graph_takes(dumped):
    """训练器的更新逻辑（`remember`，条件化视野覆盖更新）与计算图的更新逻辑
    （在记忆面、来源特征面与视野掩码上执行 `Where`）必须是同一函数。
    当前可见区域（vis=1）采用当前观测 sources，迷雾区域（vis=0）保留历史记忆 before。
    本测试在按棋盘尺寸串联的参考观测上断言这一点；第 0 回合为全零。"""
    for state, obs in chained(cases_of(dumped)):
        size = obs["size"]
        before = planes.encode_memory(state, size)
        if state is None:
            assert before.sum() == 0 and before.dtype == planes.NP_DTYPE
        after = planes.encode_memory(planes.remember(state, obs), size)
        encoded = planes.encode(obs)
        sources = encoded[:, list(planes.MEMORY_SOURCES)]
        vis = encoded[:, [planes.VIS_SOURCE]] != 0
        assert after.shape == (1, planes.N_MEMORY, *size) and after.dtype == planes.NP_DTYPE
        assert np.array_equal(after, np.where(vis, sources, before)), f"remember is not Where at {size}"


test_remember_is_the_max_the_graph_takes = test_remember_is_the_where_the_graph_takes


def test_remember_where_razed_hill_cleared_in_view_and_retained_in_fog():
    """显式验证 IDEA-002 核心假设：
    当敌丘在当前视野中被摧毁（Ot=0, Vt=1）时，历史记忆清除归零（消除幽灵残留）；
    而处于战争迷雾中的敌丘（Vt=0），原有历史记忆 Mt 完好保留。
    同时对比验证 mode='max' 依然会产生幽灵敌丘缺陷。
    """
    rows, cols = 16, 16
    b = planes.Board(rows, cols)

    # 己方蚂蚁位于 (5, 4)
    mine = [[5, 4]]
    mine_plane = b.scatter(mine)
    vis_plane = b.dilate(mine_plane, planes.VIEW_RADIUS2)

    # 验证视野覆盖：(5, 5) 必定在视野内，(12, 12) 必定在迷雾中
    assert vis_plane[5, 5] == 1, "点 (5, 5) 距离蚂蚁 (5, 4) 仅 1 格，必须在视野内 (Vt=1)"
    assert vis_plane[12, 12] == 0, "点 (12, 12) 距离蚂蚁欧氏平方 113 > 77，必须在迷雾中 (Vt=0)"

    # 将 vis_plane 编码为合法 RLE 游程
    flat = vis_plane.ravel()
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_points = [0] + change_indices.tolist() + [len(flat)]
    rle_runs = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        rle_runs.extend([int(flat[start]), end - start])

    obs = {
        "size": [rows, cols],
        "mine": mine,
        "foes": [],
        "food": [],
        "hills": [],  # 敌丘已被摧毁，无任何可见蚁穴 (Ot=0)
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": rle_runs},
    }

    # 上一回合记忆包含两个敌丘坐标
    state = {"food_seen": [], "hill_foe_seen": [[5, 5], [12, 12]]}

    # 1. 验证 mode="where"（默认）
    after_where = planes.remember(state, obs, mode="where")
    # (5, 5) 在视野内确认为空，必须被消除；(12, 12) 在迷雾中，必须保留
    assert after_where["hill_foe_seen"] == [[12, 12]], (
        f"Where 模式未能清除已探明摧毁的敌丘 (5, 5) 或丢失了迷雾记忆 (12, 12): {after_where['hill_foe_seen']}"
    )

    # 验证与张量级 np.where 逐位完全一致
    before_t = planes.encode_memory(state, [rows, cols])
    after_t = planes.encode_memory(after_where, [rows, cols])
    encoded_obs = planes.encode(obs)
    sources_t = encoded_obs[:, list(planes.MEMORY_SOURCES)]
    vis_t = encoded_obs[:, [planes.VIS_SOURCE]] != 0
    expected_t = np.where(vis_t, sources_t, before_t)
    assert np.array_equal(after_t, expected_t), "NumPy 状态编码与张量 np.where 不一致"

    # 2. 验证 mode="max"（对照消融）
    after_max = planes.remember(state, obs, mode="max")
    # max 模式因 max(1, 0) = 1，必定保留 (5, 5) 幽灵残留
    assert after_max["hill_foe_seen"] == [[5, 5], [12, 12]], (
        "Max 模式应保留全部历史坐标（包含幽灵敌丘）"
    )
    assert np.array_equal(planes.encode_memory(after_max, [rows, cols]), np.maximum(before_t, sources_t))


def test_remember_mode_max_ablation():
    """验证 mode='max' 严格维持集合单调并集行为，供 EXP-010 / EXP-012 消融实验使用。
    断言无论视野如何变化，只要出现过的坐标均持续累加保留，且与 np.maximum 完全逐位等价。"""
    rows, cols = 16, 16
    b = planes.Board(rows, cols)

    # 初始状态：记录过食物 (2, 2) 与敌丘 (4, 4)
    state = {"food_seen": [[2, 2]], "hill_foe_seen": [[4, 4]]}

    # 当前观测：蚂蚁在全图视野下，食物被采完（无食物），敌丘已摧毁（无丘），但新增食物 (7, 7)
    obs = {
        "size": [rows, cols],
        "mine": [[0, 0]],
        "foes": [],
        "food": [[7, 7]],
        "hills": [],
        "water": {"rle": [0, rows * cols]},
        "vis": {"rle": [1, rows * cols]},  # 全图可见 (Vt = 1)
    }

    before = planes.encode_memory(state, [rows, cols])
    after_max = planes.remember(state, obs, mode="max")
    after_max_encoded = planes.encode_memory(after_max, [rows, cols])

    sources = planes.encode(obs)[:, list(planes.MEMORY_SOURCES)]
    expected_max = np.maximum(before, sources)

    # 严格断言 max 行为
    assert np.array_equal(after_max_encoded, expected_max), "mode='max' 未能与 np.maximum 保持一致"
    assert [2, 2] in after_max["food_seen"] and [7, 7] in after_max["food_seen"]
    assert [4, 4] in after_max["hill_foe_seen"]

    # 对比断言 where 行为必定产生差异（在全图可见下，已清空的点不应保留）
    after_where = planes.remember(state, obs, mode="where")
    assert after_where["food_seen"] == [[7, 7]], "where 模式在全图可见下应只保留当前可见食物"
    assert after_where["hill_foe_seen"] == [], "where 模式在全图可见下应彻底清除被摧毁敌丘"
    assert not np.array_equal(after_max_encoded, planes.encode_memory(after_where, [rows, cols]))


def test_with_memory_graph_matches_numpy_where_and_max():
    """无需外部 dumped 依赖，直接测试 nets.WithMemory 的 write 与 np.where/np.maximum 的逐位一致性。"""
    torch = pytest.importorskip("torch")
    from tb_baselines import nets
    spec = {"arch": "trunk", "channels": 9, "blocks": 3, "dilate": True}
    m_where = nets.build(spec, memory="where")
    m_max = nets.build(spec, memory="max")
    assert m_where.memory_kind == "where"
    assert m_max.memory_kind == "max"

    b = torch.randint(0, 2, (1, 7, 32, 32), dtype=torch.int8)
    m = torch.randint(0, 2, (1, 2, 32, 32), dtype=torch.int8)
    out_where = m_where.write(None, b, m, mode="where")
    np_where = np.where(b[:, 6:7].numpy() != 0, b[:, [2, 5]].numpy(), m.numpy()).astype(np.int8)
    assert np.array_equal(out_where.numpy(), np_where)

    out_max = m_max.write(None, b, m, mode="max")
    np_max = np.maximum(m.numpy(), b[:, [2, 5]].numpy()).astype(np.int8)
    assert np.array_equal(out_max.numpy(), np_max)


@pytest.fixture(scope="session")
def memory_graph(tmp_path_factory) -> Path:
    """一个包含记忆清单四个端口的计算图，供 `adapt` 进行清单加载与类型校验。

    `adapt` 不会在计算图中实际执行前向推演，但会针对计算图类型检查清单声明的张量形状，
    因此端口必须存在。由于仓库未提交预训练的带记忆模型，此处使用 14 行代码通过 `onnx.helper`
    临时构造图：策略输出为对棋盘切片的 5 个通道并转为 float；记忆输出则直接直通输入。
    该测试检验的是适配器，而非计算图自身。"""
    onnx = pytest.importorskip("onnx", reason="the memory adapter test builds its graph with onnx")
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    b, m = planes.N_PLANES, planes.N_MEMORY
    g = h.make_graph(
        [h.make_node("Cast", ["board"], ["bf"], to=T.FLOAT),
         h.make_node("Slice", ["bf", "s0", "e5", "ax1"], ["policy"]),
         h.make_node("Identity", ["memory_in"], ["memory"])],
        "memory-ports",
        [h.make_tensor_value_info("board", T.INT8, [1, b, "H", "W"]),
         h.make_tensor_value_info("memory_in", T.INT8, [1, m, "H", "W"])],
        [h.make_tensor_value_info("policy", T.FLOAT, [1, planes.N_MOVES, "H", "W"]),
         h.make_tensor_value_info("memory", T.INT8, [1, m, "H", "W"])],
        [nh.from_array(np.array([0], dtype=np.int64), "s0"),
         nh.from_array(np.array([planes.N_MOVES], dtype=np.int64), "e5"),
         nh.from_array(np.array([1], dtype=np.int64), "ax1")],
    )
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("graph") / "memory-ports.onnx"
    onnx.save(model, str(path))
    return path


def test_the_memory_adapter_hands_back_the_memory_it_is_given_and_zeros_without(dumped, memory_graph, tmp_path):
    """天梯对 `memory_in` 的官方处理逻辑：通过 datalogic，携带记忆的视野将该记忆产出为计算图待读取的张量，
    未携带记忆的视野则产出全零张量，且旁侧的棋盘适配器保持不变。
    观测文件将记忆作为嵌套数组携带，`adapt` 会将其解码为清单中为 `memory` 声明的 dtype，
    正如手册中《记忆》页面所述。"""
    views = []
    for i, (state, obs) in enumerate(chained(cases_of(dumped), per_size=3)):
        view = dict(obs)
        if i % 2:
            view["memory"] = planes.encode_memory(state or planes.remember(None, obs), obs["size"]).tolist()
        views.append(view)
    assert any("memory" in v for v in views) and any("memory" not in v for v in views)

    out = tmp_path / "tensors"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(adapters.dumps(memory=True))
    (tmp_path / "views.json").write_text(json.dumps({"observations": views}))
    r = subprocess.run(
        [CLI, "adapt", str(memory_graph), str(manifest), "--obs", str(tmp_path / "views.json"),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "tinybrains adapt failed on a view carrying `memory`; a CLI without the memory carry "
        f"(before 0.3.0's successor) cannot run this test:\n{r.stdout}\n{r.stderr}"
    )
    index = json.loads((out / "index.json").read_text())
    assert len(index["cases"]) == len(views)
    for case in index["cases"]:
        i = case["case"]
        view = json.loads((out / f"case-{i}" / "observation.json").read_text())
        theirs = np.load(out / f"case-{i}" / "memory_in.npy")
        ours = planes.memory_in_view(view)
        assert theirs.shape == ours.shape and theirs.dtype == ours.dtype, (
            f"case {i}: the adapter builds {theirs.shape} {theirs.dtype}, the trainer {ours.shape} {ours.dtype}"
        )
        assert np.array_equal(theirs, ours), (
            f"case {i} ({'with' if 'memory' in view else 'without'} a memory): memory_in differs in "
            f"{int((theirs != ours).sum())} cells"
        )
        assert np.array_equal(np.load(out / f"case-{i}" / "board.npy"), planes.encode(view)), (
            f"case {i}: the board adapter changed beside the memory input"
        )


def test_the_graphs_memory_update_matches_the_encoder(dumped):
    """`nets.WithMemory` 将 `Where(vis, board[sources], memory_in)` 写入为 `i8`；训练器的 `remember`
    必须为下一回合生成完全相同的特征面。Torch 是开发依赖项且 CI 并不安装，
    因此 CI 依赖 `test_remember_is_the_where_the_graph_takes`；在执行模型导出前请运行本测试。"""
    torch = pytest.importorskip("torch")
    from tb_baselines import nets
    net = nets.build({"arch": "trunk", "channels": 4, "blocks": 1}, memory=True).eval()
    for state, obs in chained(cases_of(dumped), per_size=6):
        size = obs["size"]
        mem_in = planes.encode_memory(state, size)
        with torch.no_grad():
            policy, mem_out = net(torch.from_numpy(planes.encode(obs)), torch.from_numpy(mem_in))
        assert policy.shape == (1, planes.N_MOVES, *size)
        assert mem_out.dtype == torch.int8
        want = planes.encode_memory(planes.remember(state, obs), size)
        assert np.array_equal(mem_out.numpy(), want), f"the graph and the encoder disagree at {size}"


# ---- 单蚁记忆 -------------------------------------------------------------------

@pytest.fixture(scope="session")
def ant_graph(tmp_path_factory) -> Path:
    """一个包含单蚁记忆清单所需端口的计算图（输入：`board`、`ant_planes`、`ids`、`cells`；
    输出：`policy`、`ant_memory`），供 `adapt` 进行清单类型检查。写入的行包含 ID 与两份单元格索引副本，
    这足以满足端口形状要求；测试衡量的是适配器的联结操作，而非计算图自身。"""
    onnx = pytest.importorskip("onnx", reason="the per-ant adapter test builds its graph with onnx")
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    b, k = planes.N_PLANES, planes.ANT_MEMORY
    g = h.make_graph(
        [h.make_node("Cast", ["board"], ["bf"], to=T.FLOAT),
         h.make_node("Slice", ["bf", "s0", "e5", "ax1"], ["policy"]),
         h.make_node("Cast", ["ids"], ["idu"], to=T.UINT8),
         h.make_node("Cast", ["cells"], ["cu"], to=T.UINT8),
         h.make_node("Unsqueeze", ["idu", "ax2"], ["id3"]),
         h.make_node("Unsqueeze", ["cu", "ax2"], ["c3"]),
         h.make_node("Concat", ["id3", "c3", "c3"], ["ant_memory"], axis=2)],
        "ant-ports",
        [h.make_tensor_value_info("board", T.INT8, [1, b, "H", "W"]),
         h.make_tensor_value_info("ant_planes", T.UINT8, [1, k, "H", "W"]),
         h.make_tensor_value_info("ids", T.INT64, [1, "N"]),
         h.make_tensor_value_info("cells", T.INT64, [1, "N"])],
        [h.make_tensor_value_info("policy", T.FLOAT, [1, planes.N_MOVES, "H", "W"]),
         h.make_tensor_value_info("ant_memory", T.UINT8, [1, "N", k + 1])],
        [nh.from_array(np.array([0], dtype=np.int64), "s0"),
         nh.from_array(np.array([planes.N_MOVES], dtype=np.int64), "e5"),
         nh.from_array(np.array([1], dtype=np.int64), "ax1"),
         nh.from_array(np.array([2], dtype=np.int64), "ax2")],
    )
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("graph") / "ant-ports.onnx"
    onnx.save(model, str(path))
    return path


def test_the_per_ant_manifest_declares_the_three_inputs_and_one_row_an_ant():
    doc = json.loads(adapters.dumps(ants=True))
    assert [i["name"] for i in doc["inputs"]] == ["board", "ant_planes", "ids", "cells"]
    assert [o["name"] for o in doc["outputs"]] == ["policy", "ant_memory"]
    assert doc["outputs"][1] == {"name": "ant_memory", "dtype": planes.ANT_DTYPE,
                                 "shape": [1, "N", planes.ANT_MEMORY + 1]}
    assert doc["probe_dims"]["N"], "the probe binds the ant axis"
    assert adapters.dumps() == adapters.dumps(ants=False), "the plain manifest is untouched"
    text = json.dumps(doc["inputs"][1]["adapter"])
    assert '"gather"' in text and '"scatter"' in text, "rows re-keyed by id, then written at cells"


def test_the_ant_join_reads_last_turns_rows_back_by_id_and_zeros_for_a_new_ant():
    """纯 numpy 渲染测试，在构造的回合数据上验证：上一回合蚂蚁编号为 7, 3, 12，
    本回合为 3, 12, 20（手册中的经典示例），因此读回的行对应 3、12 的记忆与新蚂蚁的全零。"""
    prev = [[[7, 10, 11], [3, 30, 31], [12, 120, 121]]]
    obs = {"size": [8, 8], "mine": [[1, 1], [2, 5], [6, 3]], "ids": [3, 12, 20], "ant_memory": prev}
    rows = planes.ant_rows_in_view(obs)
    assert rows.tolist() == [[30, 31], [120, 121], [0, 0]] and rows.dtype == planes.ANT_NP_DTYPE
    p = planes.ant_planes_in_view(obs)
    assert p.shape == (1, planes.ANT_MEMORY, 8, 8)
    assert p[0, :, 1, 1].tolist() == [30, 31] and p[0, :, 2, 5].tolist() == [120, 121]
    assert p[0, :, 6, 3].tolist() == [0, 0] and int(p.sum()) == 30 + 31 + 120 + 121
    assert planes.ant_ids_in_view({"ids": [3, 300]}).tolist() == [[3, 300 % planes.ANT_TABLE]]
    assert planes.ant_cells_in_view(obs).tolist() == [[9, 21, 51]]


def test_the_per_ant_adapter_joins_by_id_as_the_trainer_does(dumped, ant_graph, tmp_path):
    """天梯通过 datalogic 的官方执行验证：携带上一回合 `ant_memory` 的视野 ——
    即为相同棋盘尺寸的上一个观测中存在的蚂蚁 ID 记录的行数据，因此本回合的部分蚂蚁能找到记忆行，
    而部分为新生蚂蚁 —— 产出与训练器构建完全一致的 `ant_planes`、`ids` 和 `cells`，
    未携带记忆的视野则产出全零。"""
    views = []
    prev_ids: dict[tuple[int, int], list[int]] = {}
    for obs in cases_of(dumped):
        view = dict(obs)
        key = tuple(obs["size"])
        before = prev_ids.get(key)
        if before:
            # 为该尺寸上一个视野中的每只蚂蚁保留一行，值以其 ID 编码；
            # 以及一行对应早已消亡的蚂蚁 ID，本回合任何蚂蚁都不应读取到该行。
            view["ant_memory"] = [[[i % planes.ANT_TABLE, (i * 7) % 256, (i * 13) % 256] for i in before]
                                  + [[199, 5, 6]]]
        views.append(view)
        prev_ids[key] = list(obs["ids"])
    assert any("ant_memory" in v for v in views) and any("ant_memory" not in v for v in views)

    out = tmp_path / "tensors"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(adapters.dumps(ants=True))
    (tmp_path / "views.json").write_text(json.dumps({"observations": views}))
    r = subprocess.run(
        [CLI, "adapt", str(ant_graph), str(manifest), "--obs", str(tmp_path / "views.json"),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"tinybrains adapt failed on a view carrying `ant_memory`:\n{r.stdout}\n{r.stderr}"
    index = json.loads((out / "index.json").read_text())
    assert len(index["cases"]) == len(views)
    joined = 0
    for case in index["cases"]:
        i = case["case"]
        view = json.loads((out / f"case-{i}" / "observation.json").read_text())
        for name, ours in (("ant_planes", planes.ant_planes_in_view(view)),
                           ("ids", planes.ant_ids_in_view(view)),
                           ("cells", planes.ant_cells_in_view(view))):
            theirs = np.load(out / f"case-{i}" / f"{name}.npy")
            assert theirs.shape == ours.shape and theirs.dtype == ours.dtype, (
                f"case {i} {name}: the adapter builds {theirs.shape} {theirs.dtype}, the trainer {ours.shape} {ours.dtype}")
            assert np.array_equal(theirs, ours), f"case {i} {name}: {int((theirs != ours).sum())} elements differ"
        joined += int(planes.ant_planes_in_view(view).any())
        assert np.array_equal(np.load(out / f"case-{i}" / "board.npy"), planes.encode(view))
    assert joined, "至少有一个视野通过关联操作读回了记忆行"
