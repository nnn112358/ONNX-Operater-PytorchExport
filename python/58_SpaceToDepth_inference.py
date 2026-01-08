import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("58_space_to_depth.onnx")
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("SpaceToDepth演算が正常に実行されました")
