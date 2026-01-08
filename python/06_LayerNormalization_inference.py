import onnxruntime as ort
import numpy as np

# ONNXモデルを読み込んで推論セッションを作成
session = ort.InferenceSession("06_layer_normalization.onnx")

# 入力データを準備（バッチサイズ4、128次元の特徴ベクトル）
input_data = np.random.randn(4, 128).astype(np.float32)

# 推論を実行
inputs = {"input": input_data}
result = session.run(["output"], inputs)[0]

print(f"入力形状: {input_data.shape}")
print(f"出力形状: {result.shape}")
print("レイヤー正規化が正常に実行されました")
print(f"入力の最初のサンプルの最初の5要素: {input_data[0, :5]}")
print(f"出力の最初のサンプルの最初の5要素: {result[0, :5]}")
