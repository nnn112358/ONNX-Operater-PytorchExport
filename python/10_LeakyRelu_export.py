
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class LeakyReluModel(nn.Module):
    """
    Leaky ReLU活性化関数を持つニューラルネットワークモデル

    Leaky ReLU: x if x > 0 else negative_slope * x
    負の値に小さな勾配を持たせることで、dying ReLU問題を軽減します。
    """

    def __init__(self, negative_slope=0.01):
        """
        モデルの初期化

        Args:
            negative_slope: 負の入力に対する傾き
        """
        super(LeakyReluModel, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            Leaky ReLU活性化関数を適用した出力テンソル
        """
        return self.leaky_relu(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = LeakyReluModel(negative_slope=0.01)

# ダミー入力の作成
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "10_leaky_relu.onnx",           # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 10_leaky_relu.onnx")
