import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("14_softmax.onnx")
input_data = np.random.randn(1, 10).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("Softmax活性化関数が正常に実行されました")
print(f"入力: {input_data[0]}")
print(f"出力: {result[0]}")
print(f"出力の合計（確率の合計=1）: {result[0].sum()}")
