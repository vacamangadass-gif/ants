import os
import numpy as np
import onnx
import onnxruntime as ort

model_path = "models/exp084/model.onnx"

# 1. 物理文件体积检查
file_size = os.path.getsize(model_path)
assert file_size <= 131072, f"模型体积超过 128 KB 限制: {file_size} > 131072"
print(f"[通过] 物理大小验证: {file_size} 字节 <= 131,072 字节 (128 KB)")

# 2. ONNX 规范检验
model = onnx.load(model_path)
onnx.checker.check_model(model)
print("[通过] ONNX 结构合规检验")

# 3. 输入输出张量签名与 Critic 剥离检验
input_dict = {inp.name: inp for inp in model.graph.input}
output_dict = {out.name: out for out in model.graph.output}
print(f"输入列表: {list(input_dict.keys())}")
print(f"输出列表: {list(output_dict.keys())}")

assert set(input_dict.keys()) == {"board", "memory_in"}, f"输入签名不符合规范: {list(input_dict.keys())}"
assert set(output_dict.keys()) == {"policy", "memory"}, f"输出签名不符合规范: {list(output_dict.keys())}"

# 严查是否包含任何价值评估、Critic、特权状态相关名称
forbidden_tokens = ["value", "critic", "v_head", "vhead", "privileged", "reward", "baseline"]
for tensor in list(model.graph.initializer) + list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
    name_lower = tensor.name.lower()
    for tok in forbidden_tokens:
        assert tok not in name_lower, f"图中发现违规 Critic 残留张量: {tensor.name}"

for node in model.graph.node:
    node_name_lower = node.name.lower()
    for tok in forbidden_tokens:
        assert tok not in node_name_lower, f"图中发现违规 Critic 残留节点: {node.name}"

# 算子集合检验：Critic 全连接层所使用的 Gemm/MatMul 严禁存在
op_types = sorted(list({node.op_type for node in model.graph.node}))
print(f"算子集合: {op_types}")
assert "Gemm" not in op_types, "发现 Gemm 算子，存在未剥离全连接层"
assert "MatMul" not in op_types, "发现 MatMul 算子，存在未剥离全连接层"

# 4. 参数量核算
init_params = sum(int(np.prod(list(init.dims))) for init in model.graph.initializer)
const_params = 0
for node in model.graph.node:
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                t = attr.t
                dims = list(t.dims)
                const_params += int(np.prod(dims)) if dims else 1

print(f"初始化张量参数: {init_params}, 常量节点参数: {const_params}, 总计: {init_params + const_params}")
assert (init_params + const_params) == 24577 or init_params == 24507, f"参数量异常: init={init_params}, const={const_params}"
print("[通过] 参数量核算一致")

# 5. ONNX Runtime 边界条件推理与输出数值合法性测试
session = ort.InferenceSession(model_path)

test_dims = [(24, 24), (32, 32), (45, 55), (60, 60), (124, 124)]
for h, w in test_dims:
    # 正常全零输入
    board_zero = np.zeros((1, 7, h, w), dtype=np.int8)
    mem_zero = np.zeros((1, 2, h, w), dtype=np.int8)
    out_p, out_m = session.run(None, {"board": board_zero, "memory_in": mem_zero})
    assert out_p.shape == (1, 5, h, w), f"尺寸 ({h}, {w}) 策略输出形状异常: {out_p.shape}"
    assert out_m.shape == (1, 2, h, w), f"尺寸 ({h}, {w}) 记忆输出形状异常: {out_m.shape}"
    assert not np.isnan(out_p).any(), f"策略输出出现 NaN ({h}, {w})"
    assert not np.isinf(out_p).any(), f"策略输出出现 Inf ({h}, {w})"
    assert not np.isnan(out_m).any(), f"记忆输出出现 NaN ({h}, {w})"

    # 极端值输入边界测试
    board_extreme = np.random.choice([np.int8(-128), np.int8(127)], size=(1, 7, h, w))
    mem_extreme = np.random.choice([np.int8(-128), np.int8(127)], size=(1, 2, h, w))
    out_p_ext, out_m_ext = session.run(None, {"board": board_extreme, "memory_in": mem_extreme})
    assert not np.isnan(out_p_ext).any(), f"极端输入下策略输出出现 NaN ({h}, {w})"
    assert not np.isinf(out_p_ext).any(), f"极端输入下策略输出出现 Inf ({h}, {w})"

print("[通过] 多尺度网格与极端数值推演边界测试全量通过")
