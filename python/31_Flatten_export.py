
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class FlattenModel(nn.Module):
    """
    テンソルを平坦化（Flatten）するモデル

    このモデルは多次元テンソルを指定次元から1次元に平坦化します。
    演算: 指定次元以降を1次元に結合

    用途:
    - CNN層の出力を全結合層に接続する際の前処理
    - 画像データをベクトルに変換
    - バッチ処理を保持しながらその他の次元を平坦化

    注意:
    - start_dim以前の次元は保持される
    - デフォルトではバッチ次元（0次元目）は保持される
    """

    def __init__(self, start_dim=1):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、平坦化層を定義する

        Args:
            start_dim: 平坦化を開始する次元（デフォルト: 1、バッチ次元は保持）
        """
        super(FlattenModel, self).__init__()
        # Flatten層を定義
        # - start_dim=1: バッチ次元（0次元目）を保持し、それ以降を平坦化
        self.flatten = nn.Flatten(start_dim=start_dim)

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            平坦化されたテンソル
            例: 入力(1, 3, 32, 32)の場合、出力は(1, 3072)
        """
        return self.flatten(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - start_dim=1: バッチ次元を保持し、チャンネル以降を平坦化
model = FlattenModel(start_dim=1)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 平坦化後: (1, 3*32*32) = (1, 3072)
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "31_flatten.onnx",          # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 31_flatten.onnx")
