
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SpaceToDepthModel(nn.Module):
    """
    空間次元をチャンネル次元に変換するモデル

    このモデルは空間情報をチャンネル情報に再編成します。
    演算: HxW空間を減らしてチャンネル数を増やす

    用途:
    - 解像度の削減とチャンネル数の増加
    - 情報損失なしのダウンサンプリング
    - 物体検出モデル（YOLOなど）での利用

    注意:
    - block_sizeは入力の高さと幅を均等に分割できる必要がある
    - チャンネル数はblock_size²倍になる
    """

    def __init__(self, block_size=2):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、ブロックサイズを設定する

        Args:
            block_size: ブロックサイズ（デフォルト: 2）
                       空間をblock_size x block_sizeのブロックに分割
        """
        super(SpaceToDepthModel, self).__init__()
        # ブロックサイズを保存
        self.block_size = block_size

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            変換されたテンソル
            例: 入力(1, 3, 32, 32)でblock_size=2の場合、出力は(1, 12, 16, 16)
                チャンネル: 3 * 2² = 12、サイズ: 32/2 = 16
        """
        # PyTorchのpixel_unshuffle（SpaceToDepthと同等）
        return torch.nn.functional.pixel_unshuffle(x, self.block_size)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - block_size=2: 2x2のブロックに分割
model = SpaceToDepthModel(block_size=2)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 変換後: (1, 12, 16, 16)
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "58_space_to_depth.onnx",   # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 58_space_to_depth.onnx")
