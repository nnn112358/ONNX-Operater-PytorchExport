
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class UnsqueezeModel(nn.Module):
    """
    サイズ1の次元を追加（Unsqueeze）するモデル

    このモデルは指定位置にサイズ1の新しい次元を挿入します。
    演算: 指定位置に新しい次元を追加して次元数を増やす

    用途:
    - バッチ次元の追加
    - ブロードキャストのための次元調整
    - チャンネル次元の追加

    注意:
    - 負のインデックスも使用可能（-1は最後の次元の後）
    - 挿入後の次元インデックスは元の次元より増加する
    """

    def __init__(self, dim):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、追加する次元位置を設定する

        Args:
            dim: 新しい次元を挿入する位置のインデックス
        """
        super(UnsqueezeModel, self).__init__()
        # 追加する次元の位置を保存
        self.dim = dim

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, height, width)）

        Returns:
            サイズ1の次元が追加されたテンソル
            例: 入力(1, 32, 32)でdim=1の場合、出力は(1, 1, 32, 32)
        """
        return x.unsqueeze(self.dim)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim=1: 1次元目の位置に新しい次元を挿入（チャンネル次元を追加）
model = UnsqueezeModel(dim=1)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 32, 32): バッチサイズ1、32x32ピクセル（チャンネル次元なし）
# - unsqueeze後: (1, 1, 32, 32)
dummy_input = torch.randn(1, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "33_unsqueeze.onnx",        # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 33_unsqueeze.onnx")
