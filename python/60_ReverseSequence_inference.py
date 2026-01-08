import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("60_reverse_sequence.onnx")
input_data = np.random.randn(2, 5, 10).astype(np.float32)
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("ReverseSequence演算が正常に実行されました")
print(f"入力の最初のバッチの最初の次元: {input_data[0, :, 0]}")
print(f"出力の最初のバッチの最初の次元（反転）: {result[0, :, 0]}")
