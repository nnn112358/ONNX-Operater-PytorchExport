import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("29_reshape.onnx")
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("Reshape演算が正常に実行されました")
print(f"要素数: 入力={input_data.size}, 出力={result.size}")
