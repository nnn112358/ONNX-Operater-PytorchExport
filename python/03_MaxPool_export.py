
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class MaxPoolModel(nn.Module):
    """
    最大値プーリング層を持つニューラルネットワークモデル

    最大値プーリングは、局所領域の最大値を取ることで特徴マップを縮小します。
    入力: 3チャンネルの特徴マップ
    出力: 空間次元が半分になった特徴マップ
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、MaxPooling層を定義する
        """
        super(MaxPoolModel, self).__init__()

        # 2D最大値プーリング層の定義
        # - kernel_size=2   : プーリングウィンドウのサイズ（2x2）
        # - stride=2        : ストライド（2ピクセルずつ移動）
        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            最大値プーリング層を通過した出力テンソル
            空間次元が半分になる
        """
        return self.maxpool(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = MaxPoolModel()

# ダミー入力の作成（3チャンネル、32x32の特徴マップ）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "03_maxpool.onnx",              # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 03_maxpool.onnx")
