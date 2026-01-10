
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SigmoidModel(nn.Module):
    """
    Sigmoid活性化関数を持つニューラルネットワークモデル

    Sigmoid: 1 / (1 + exp(-x))
    出力を0から1の範囲に変換します。
    二値分類問題の出力層や、ゲート機構でよく使用されます。
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、Sigmoid活性化関数を定義する
        """
        super(SigmoidModel, self).__init__()
        # Sigmoid活性化関数の定義
        # 出力を0から1の範囲に正規化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            Sigmoid活性化関数を適用した出力テンソル（0から1の範囲）
        """
        return self.sigmoid(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = SigmoidModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32):
#   - 1  : バッチサイズ（1つのサンプル）
#   - 3  : チャンネル数（RGB画像を想定）
#   - 32 : 画像の高さ（ピクセル）
#   - 32 : 画像の幅（ピクセル）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
# ONNX (Open Neural Network Exchange) は異なるフレームワーク間で
# モデルを共有するための標準フォーマット
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "15_sigmoid.onnx",          # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
                                # （TorchDynamoベースのエクスポートは使用しない）
)

# エクスポート完了メッセージを表示
print("saved: 15_sigmoid.onnx")
