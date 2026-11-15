import os
import numpy as np
import onnx
import onnxruntime as ort

model_path = "models/exp084/model.onnx"
file_size = os.path.getsize(model_path)
assert file_size <= 131072, f"文件大小超过 128KB 限制: {file_size} > 131072"
print(f"ONNX 文件体积验证通过: {file_size} 字节 <= 131072 字节")

model = onnx.load(model_path)
onnx.checker.check_model(model)
print("ONNX 结构完整性校验通过")

input_names = [inp.name for inp in model.graph.input]
output_names = [out.name for out in model.graph.output]
print(f"输入名称: {input_names}")
print(f"输出名称: {output_names}")

assert "policy" in output_names, "输出缺少 policy"
assert "memory" in output_names, "输出缺少 memory"
assert len(output_names) == 2, f"输出数量不为 2: {output_names}"

forbidden_keywords = ["value", "critic", "val", "v_head"]
for out in model.graph.output:
    for kw in forbidden_keywords:
        assert kw not in out.name.lower(), f"输出包含 Critic 关键词: {out.name}"

op_types = sorted(list({node.op_type for node in model.graph.node}))
print(f"算子集合: {op_types}")

session = ort.InferenceSession(model_path)
dummy_board = np.zeros((1, 7, 32, 32), dtype=np.int8)
dummy_memory = np.zeros((1, 2, 32, 32), dtype=np.int8)

outputs = session.run(None, {"board": dummy_board, "memory_in": dummy_memory})
policy_out, memory_out = outputs[0], outputs[1]
print(f"单步推理 policy 形状: {policy_out.shape}, 数据类型: {policy_out.dtype}")
print(f"单步推理 memory 形状: {memory_out.shape}, 数据类型: {memory_out.dtype}")

assert policy_out.shape == (1, 5, 32, 32), f"策略输出形状不符合预期: {policy_out.shape}"
assert memory_out.shape == (1, 2, 32, 32), f"记忆输出形状不符合预期: {memory_out.shape}"
print("ONNX 模型结构与推理验证全部通过")
