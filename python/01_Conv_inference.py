import onnxruntime as ort
import numpy as np

# ONNXモデルを読み込んで推論セッションを作成
session = ort.InferenceSession("01_conv.onnx")

# 入力データを準備（バッチサイズ1、3チャンネル、32x32の画像）
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)

# 推論を実行
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("畳み込み演算が正常に実行されました")
print(f"出力の一部: {result[0, 0, :3, :3]}")  # 最初のチャンネルの左上3x3を表示
