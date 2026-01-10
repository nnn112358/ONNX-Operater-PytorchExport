
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class DepthToSpaceModel(nn.Module):
    """
    チャンネル次元を空間次元に変換するモデル

    このモデルはチャンネル情報を空間情報に再編成します。
    演算: チャンネル数を減らしてHxW空間を増やす

    用途:
    - アップサンプリング（超解像など）
    - チャンネル数の削減と解像度の向上
    - ピクセルシャッフル操作

    注意:
    - チャンネル数はblock_size²で割り切れる必要がある
    - 空間サイズはblock_size倍になる
    """

    def __init__(self, block_size=2):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、ブロックサイズを設定する

        Args:
            block_size: ブロックサイズ（デフォルト: 2）
                       block_size x block_sizeのブロックを空間に展開
        """
        super(DepthToSpaceModel, self).__init__()
        # ブロックサイズを保存
        self.block_size = block_size

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            変換されたテンソル
            例: 入力(1, 12, 16, 16)でblock_size=2の場合、出力は(1, 3, 32, 32)
                チャンネル: 12 / 2² = 3、サイズ: 16 * 2 = 32
        """
        # PyTorchのpixel_shuffle（DepthToSpaceと同等）
        return torch.nn.functional.pixel_shuffle(x, self.block_size)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - block_size=2: 2x2のブロックに展開
model = DepthToSpaceModel(block_size=2)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 12, 16, 16): バッチサイズ1、12チャンネル、16x16ピクセル
# - 変換後: (1, 3, 32, 32)
dummy_input = torch.randn(1, 12, 16, 16)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "59_depth_to_space.onnx",   # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 59_depth_to_space.onnx")
