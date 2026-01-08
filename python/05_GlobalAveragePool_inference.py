import onnxruntime as ort
import numpy as np

# ONNXモデルを読み込んで推論セッションを作成
session = ort.InferenceSession("05_global_averagepool.onnx")

# 入力データを準備（バッチサイズ1、3チャンネル、32x32の特徴マップ）
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)

# 推論を実行
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("グローバル平均プーリング演算が正常に実行されました")
print(f"出力: {result.reshape(-1)}")  # 全ての出力値を表示
