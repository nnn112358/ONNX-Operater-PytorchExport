import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("27_sqrt.onnx")
input_data = np.abs(np.random.randn(1, 3, 32, 32)).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("Sqrt演算が正常に実行されました")
print(f"入力の一部: {input_data[0, 0, 0, :5]}")
print(f"出力の一部（√x）: {result[0, 0, 0, :5]}")
