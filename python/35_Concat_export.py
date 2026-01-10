
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ConcatModel(nn.Module):
    """
    複数のテンソルを連結（Concatenate）するモデル

    このモデルは指定軸に沿って複数のテンソルを結合します。
    演算: 指定次元に沿ってテンソルを連結

    用途:
    - スキップ接続（U-Netなど）
    - 複数の特徴マップの統合
    - アンサンブル学習での結果統合

    注意:
    - 連結軸以外の次元のサイズは一致している必要がある
    - 連結軸のサイズは異なっていても良い
    """

    def __init__(self, dim=1):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、連結軸を設定する

        Args:
            dim: 連結する軸のインデックス（デフォルト: 1、チャンネル軸）
        """
        super(ConcatModel, self).__init__()
        # 連結する軸を保存
        self.dim = dim

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1の入力テンソル、形状は (batch_size, channels, height, width)
            y: 第2の入力テンソル、形状はxと同じ（連結軸以外）

        Returns:
            連結されたテンソル
            例: 入力(1, 3, 32, 32)2つをdim=1で連結すると、出力は(1, 6, 32, 32)
        """
        return torch.cat([x, y], dim=self.dim)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim=1: チャンネル軸で連結
model = ConcatModel(dim=1)

# ダミー入力1の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input1 = torch.randn(1, 3, 32, 32)

# ダミー入力2の作成
# - 入力1と同じ形状のテンソルを生成
dummy_input2 = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),   # 複数の入力をタプルで指定
    "35_concat.onnx",               # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"], # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 35_concat.onnx")
