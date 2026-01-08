import onnxruntime as ort
import numpy as np

# ONNXモデルを読み込んで推論セッションを作成
session = ort.InferenceSession("10_leaky_relu.onnx")

# 入力データを準備（-1から1の範囲のランダムな値）
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)

# 推論を実行
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("Leaky ReLU活性化関数が正常に実行されました")
print(f"入力の一部: {input_data[0, 0, 0, :5]}")
print(f"出力の一部: {result[0, 0, 0, :5]}")
