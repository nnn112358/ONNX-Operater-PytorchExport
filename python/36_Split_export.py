
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SplitModel(nn.Module):
    """
    テンソルを分割（Split）するモデル

    このモデルは指定軸に沿ってテンソルを複数に分割します。
    演算: 指定次元に沿ってテンソルを複数の部分に分ける

    用途:
    - 特徴マップの分岐
    - マルチヘッドアテンションでの分割
    - モデルの並列処理

    注意:
    - 分割サイズの合計は元のテンソルのサイズと一致する必要がある
    - 均等分割または不均等分割が可能
    """

    def __init__(self, split_size, dim=1):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、分割パラメータを設定する

        Args:
            split_size: 各分割のサイズ（整数またはリスト）
            dim: 分割する軸のインデックス（デフォルト: 1、チャンネル軸）
        """
        super(SplitModel, self).__init__()
        # 分割サイズを保存
        self.split_size = split_size
        # 分割する軸を保存
        self.dim = dim

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            分割されたテンソルのタプル
            例: 入力(1, 4, 32, 32)をdim=1、split_size=2で分割すると、
                2つの(1, 2, 32, 32)テンソルを返す
        """
        splits = torch.split(x, self.split_size, dim=self.dim)
        return splits[0], splits[1]


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - split_size=2: 各分割のサイズを2に設定
# - dim=1: チャンネル軸で分割
model = SplitModel(split_size=2, dim=1)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 4, 32, 32): バッチサイズ1、4チャンネル、32x32ピクセル
# - 分割後: 2つの(1, 2, 32, 32)テンソル
dummy_input = torch.randn(1, 4, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "36_split.onnx",                # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output1", "output2"], # ONNX グラフでの出力テンソルの名前（複数出力）
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 36_split.onnx")
