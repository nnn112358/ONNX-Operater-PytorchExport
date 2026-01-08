import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("38_gather.onnx")
input_data = np.random.randn(2, 5, 10).astype(np.float32)
indices_data = np.array([[[0, 1, 2], [3, 4, 0], [1, 2, 3], [4, 0, 1], [2, 3, 4]]]).astype(np.int64)
indices_data = np.repeat(indices_data, 2, axis=0)
inputs = {"input": input_data, "indices": indices_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"インデックス形状: {indices_data.shape}")
print(f"出力形状: {result.shape}")
print("Gather演算が正常に実行されました")
