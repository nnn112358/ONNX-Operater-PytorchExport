import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("21_mul.onnx")
input_data1 = np.random.randn(1, 3, 32, 32).astype(np.float32)
input_data2 = np.random.randn(1, 3, 32, 32).astype(np.float32)
inputs = {"input1": input_data1, "input2": input_data2}
result = session.run(["output"], inputs)[0]

print(f"入力1形状: {input_data1.shape}")
print(f"入力2形状: {input_data2.shape}")
print(f"出力形状: {result.shape}")
print("Mul演算が正常に実行されました")
print(f"入力1の一部: {input_data1[0, 0, 0, :3]}")
print(f"入力2の一部: {input_data2[0, 0, 0, :3]}")
print(f"出力の一部: {result[0, 0, 0, :3]}")
