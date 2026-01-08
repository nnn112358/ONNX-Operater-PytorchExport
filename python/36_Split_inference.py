import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("36_split.onnx")
input_data = np.random.randn(1, 4, 32, 32).astype(np.float32)
inputs = {"input": input_data}
result1, result2 = session.run(["output1", "output2"], inputs)

print(f"入力形状: {input_data.shape}")
print(f"出力1形状: {result1.shape}")
print(f"出力2形状: {result2.shape}")
print("Split演算が正常に実行されました")
