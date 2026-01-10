
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SqueezeModel(nn.Module):
    """
    サイズ1の次元を削除（Squeeze）するモデル

    このモデルは形状が1である次元を削除します。
    演算: サイズ1の次元を除去して次元数を減らす

    用途:
    - 不要な次元の削除
    - モデル間のテンソル形状の調整
    - バッチサイズ1の次元を削除してスカラー化

    注意:
    - 指定次元のサイズが1でない場合、エラーが発生
    - dimを指定しない場合、すべてのサイズ1の次元が削除される
    """

    def __init__(self, dim=None):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、削除する次元を設定する

        Args:
            dim: 削除する次元のインデックス（Noneの場合、すべてのサイズ1の次元を削除）
        """
        super(SqueezeModel, self).__init__()
        # 削除する次元を保存
        self.dim = dim

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, 1, height, width)）

        Returns:
            サイズ1の次元が削除されたテンソル
            例: 入力(1, 1, 32, 32)でdim=1の場合、出力は(1, 32, 32)
        """
        if self.dim is not None:
            return x.squeeze(self.dim)
        return x.squeeze()


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim=1: 1次元目（チャンネル次元）のサイズ1を削除
model = SqueezeModel(dim=1)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 1, 32, 32): バッチサイズ1、チャンネル1、32x32ピクセル
# - squeeze後: (1, 32, 32)
dummy_input = torch.randn(1, 1, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "32_squeeze.onnx",          # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 32_squeeze.onnx")
