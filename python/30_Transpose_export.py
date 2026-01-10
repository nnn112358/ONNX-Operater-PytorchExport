
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class TransposeModel(nn.Module):
    """
    テンソルの次元を入れ替える（Transpose）モデル

    このモデルは指定した2つの次元を入れ替えます。
    演算: 次元の順序を変更

    用途:
    - 画像データの形式変換（CHW ⇔ HWC など）
    - 行列の転置
    - バッチ処理時の次元調整

    注意:
    - 次元のインデックスは0から始まる
    - 負のインデックスも使用可能（-1は最後の次元）
    """

    def __init__(self, dim0, dim1):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、入れ替える次元を設定する

        Args:
            dim0: 入れ替える第1の次元のインデックス
            dim1: 入れ替える第2の次元のインデックス
        """
        super(TransposeModel, self).__init__()
        # 入れ替える第1の次元を保存
        self.dim0 = dim0
        # 入れ替える第2の次元を保存
        self.dim1 = dim1

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            次元が入れ替えられたテンソル
            例: 入力(1, 3, 32, 32)でdim0=2, dim1=3の場合、出力は(1, 3, 32, 32)
        """
        return x.transpose(self.dim0, self.dim1)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim0=2, dim1=3: 高さ(H)と幅(W)の次元を入れ替え
# - 入力形状 (1, 3, 32, 32) の場合、出力形状も (1, 3, 32, 32)（正方形なので同じ）
model = TransposeModel(dim0=2, dim1=3)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "30_transpose.onnx",        # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 30_transpose.onnx")
