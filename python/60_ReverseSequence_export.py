
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ReverseSequenceModel(nn.Module):
    """
    シーケンスを指定長まで反転するモデル

    このモデルは特定の軸に沿ってシーケンスを反転します。
    演算: 指定軸の要素順序を逆にする

    用途:
    - 双方向RNNの実装
    - 時系列データの逆順処理
    - データ拡張

    注意:
    - 指定軸全体が反転される
    - バッチごとに異なる長さで反転する場合は追加の実装が必要
    """

    def __init__(self, dim=1):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、反転する軸を設定する

        Args:
            dim: 反転する軸のインデックス（デフォルト: 1）
        """
        super(ReverseSequenceModel, self).__init__()
        # 反転する軸を保存
        self.dim = dim

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, sequence_length, features)）

        Returns:
            指定軸に沿って反転されたテンソル、形状は入力と同じ
            例: 入力[1,2,3,4,5]をdim=0で反転すると、出力は[5,4,3,2,1]
        """
        # PyTorchのflip関数を使用してシーケンスを反転
        return torch.flip(x, dims=[self.dim])


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim=1: 1次元目（シーケンス次元）を反転
model = ReverseSequenceModel(dim=1)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (2, 5, 10): バッチサイズ2、シーケンス長5、特徴次元10
# - 反転後: 1次元目の順序が逆になる
dummy_input = torch.randn(2, 5, 10)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "60_reverse_sequence.onnx", # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 60_reverse_sequence.onnx")
