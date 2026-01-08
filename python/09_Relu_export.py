
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ReluModel(nn.Module):
    """
    ReLU活性化関数を持つニューラルネットワークモデル

    ReLU (Rectified Linear Unit): max(0, x)
    負の値を0に、正の値をそのまま出力します。
    """

    def __init__(self):
        """
        モデルの初期化
        """
        super(ReluModel, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            ReLU活性化関数を適用した出力テンソル
        """
        return self.relu(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = ReluModel()

# ダミー入力の作成
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "09_relu.onnx",                 # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 09_relu.onnx")
