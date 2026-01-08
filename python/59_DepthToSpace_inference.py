import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("59_depth_to_space.onnx")
input_data = np.random.randn(1, 12, 16, 16).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("DepthToSpace演算が正常に実行されました")
