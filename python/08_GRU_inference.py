import onnxruntime as ort
import numpy as np

# ONNXモデルを読み込んで推論セッションを作成
session = ort.InferenceSession("08_gru.onnx")

# 入力データを準備（バッチサイズ4、シーケンス長5、特徴次元10）
input_data = np.random.randn(4, 5, 10).astype(np.float32)

# 推論を実行
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("GRU演算が正常に実行されました")
print(f"出力の一部: {result[0, 0, :5]}")  # 最初のバッチの最初のタイムステップの最初の5要素
