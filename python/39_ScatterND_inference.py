import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("39_scatter_nd.onnx")
data = np.zeros((2, 5)).astype(np.float32)
indices = np.array([[0, 1, 2], [2, 3, 4]]).astype(np.int64)
updates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)
inputs = {"data": data, "indices": indices, "updates": updates}
result = session.run(["output"], inputs)[0]

print(f"データ形状: {data.shape}")
print(f"インデックス形状: {indices.shape}")
print(f"更新値形状: {updates.shape}")
print(f"出力形状: {result.shape}")
print("ScatterND演算が正常に実行されました")
print(f"出力:\n{result}")
