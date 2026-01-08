
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class AveragePoolModel(nn.Module):
    """
    平均値プーリング層を持つニューラルネットワークモデル

    平均値プーリングは、局所領域の平均値を取ることで特徴マップを縮小します。
    入力: 3チャンネルの特徴マップ
    出力: 空間次元が半分になった特徴マップ
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、AveragePooling層を定義する
        """
        super(AveragePoolModel, self).__init__()

        # 2D平均値プーリング層の定義
        # - kernel_size=2   : プーリングウィンドウのサイズ（2x2）
        # - stride=2        : ストライド（2ピクセルずつ移動）
        self.avgpool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            平均値プーリング層を通過した出力テンソル
            空間次元が半分になる
        """
        return self.avgpool(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = AveragePoolModel()

# ダミー入力の作成（3チャンネル、32x32の特徴マップ）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "04_averagepool.onnx",          # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 04_averagepool.onnx")
