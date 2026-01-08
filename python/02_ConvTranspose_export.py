
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ConvTransposeModel(nn.Module):
    """
    転置畳み込み層を持つニューラルネットワークモデル

    転置畳み込みは逆畳み込みとも呼ばれ、画像のアップサンプリング（拡大）に使用されます。
    入力: 16チャンネルの特徴マップ
    出力: RGB画像（3チャンネル）
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、転置畳み込み層を定義する
        """
        super(ConvTransposeModel, self).__init__()

        # 2D転置畳み込み層の定義
        # - in_channels=16  : 入力チャンネル数
        # - out_channels=3  : 出力チャンネル数（RGB）
        # - kernel_size=3   : カーネルのサイズ（3x3）
        # - stride=2        : ストライド（出力を2倍に拡大）
        # - padding=1       : パディング
        # - output_padding=1: 出力パディング（出力サイズを調整）
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            転置畳み込み層を通過した出力テンソル
            空間次元が2倍に拡大される
        """
        return self.conv_transpose(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = ConvTransposeModel()

# ダミー入力の作成（16チャンネル、16x16の特徴マップ）
dummy_input = torch.randn(1, 16, 16, 16)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "02_conv_transpose.onnx",       # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 02_conv_transpose.onnx")
