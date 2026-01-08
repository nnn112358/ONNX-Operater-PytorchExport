
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ConvModel(nn.Module):
    """
    シンプルな2D畳み込み層を持つニューラルネットワークモデル
    
    このモデルは画像処理の基本的な畳み込み演算を行います。
    入力: RGB画像（3チャンネル）
    出力: 16チャンネルの特徴マップ
    """
    
    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、畳み込み層を定義する
        """
        super(ConvModel, self).__init__()
        
        # 2D畳み込み層の定義
        # - in_channels=3   : 入力チャンネル数（RGBの3チャンネル）
        # - out_channels=16 : 出力チャンネル数（16種類の特徴を抽出）
        # - kernel_size=3   : カーネル（フィルタ）のサイズ（3x3ピクセル）
        # - stride=1        : カーネルの移動幅（1ピクセルずつ移動）
        # - padding=1       : 入力の周囲に1ピクセルのパディングを追加
        #                     （出力サイズを入力サイズと同じに保つため）
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def forward(self, x):
        """
        順伝播（フォワードパス）の定義
        
        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)
               例: (1, 3, 32, 32) = バッチサイズ1、RGB3チャンネル、32x32ピクセル
        
        Returns:
            畳み込み層を通過した出力テンソル
            形状は (batch_size, 16, height, width)
        """
        return self.conv(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = ConvModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32):
#   - 1  : バッチサイズ（1枚の画像）
#   - 3  : チャンネル数（RGB）
#   - 32 : 画像の高さ（ピクセル）
#   - 32 : 画像の幅（ピクセル）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
# ONNX (Open Neural Network Exchange) は異なるフレームワーク間で
# モデルを共有するための標準フォーマット
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "01_conv.onnx",             # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
                                # （TorchDynamoベースのエクスポートは使用しない）
)

# エクスポート完了メッセージを表示
print("saved: 01_conv.onnx")