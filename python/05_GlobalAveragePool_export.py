
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class GlobalAveragePoolModel(nn.Module):
    """
    グローバル平均プーリング層を持つニューラルネットワークモデル

    グローバル平均プーリングは、各チャンネルの全空間領域にわたって平均値を計算します。
    空間次元が完全に削減され、チャンネルごとに1つの値になります。
    入力: 3チャンネルの特徴マップ
    出力: 空間次元が1x1になった特徴マップ
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、GlobalAveragePooling層を定義する
        """
        super(GlobalAveragePoolModel, self).__init__()

        # 2Dグローバル平均プーリング層
        # AdaptiveAvgPool2dを使用して出力サイズを(1,1)に固定
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            グローバル平均プーリング層を通過した出力テンソル
            形状は (batch_size, channels, 1, 1)
        """
        return self.global_avgpool(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = GlobalAveragePoolModel()

# ダミー入力の作成（3チャンネル、32x32の特徴マップ）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                              # エクスポートするPyTorchモデル
    dummy_input,                        # モデルの入力形状を推論するためのダミー入力
    "05_global_averagepool.onnx",      # 出力ファイル名（ONNX形式）
    input_names=["input"],              # ONNX グラフでの入力テンソルの名前
    output_names=["output"],            # ONNX グラフでの出力テンソルの名前
    dynamo=False                        # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 05_global_averagepool.onnx")
