
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ExpModel(nn.Module):
    """
    テンソルの指数関数（Exponential）を計算するモデル

    このモデルは自然対数の底e（約2.71828）を入力値で累乗します。
    演算: output[i] = e^input[i]

    用途:
    - Softmax関数の計算（確率分布への変換）
    - 指数分布のモデリング
    - 活性化関数としての利用

    注意:
    - 入力値が大きいと結果がオーバーフローする可能性がある
    - 数値安定性のため、大きな値を入力する場合は正規化が推奨される
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(ExpModel, self).__init__()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            指数関数が適用されたテンソル、形状は入力と同じ
        """
        return torch.exp(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = ExpModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 正規分布の値（平均0、標準偏差1）なので、オーバーフローのリスクは低い
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "25_exp.onnx",              # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 25_exp.onnx")
