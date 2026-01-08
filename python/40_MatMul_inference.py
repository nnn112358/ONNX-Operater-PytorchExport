import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("40_matmul.onnx")
input_data1 = np.random.randn(1, 4, 8).astype(np.float32)
input_data2 = np.random.randn(1, 8, 16).astype(np.float32)
inputs = {"input1": input_data1, "input2": input_data2}
result = session.run(["output"], inputs)[0]

print(f"入力1形状: {input_data1.shape}")
print(f"入力2形状: {input_data2.shape}")
print(f"出力形状: {result.shape}")
print("MatMul演算が正常に実行されました")
