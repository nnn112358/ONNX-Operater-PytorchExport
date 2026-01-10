
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ScatterNDModel(nn.Module):
    """
    インデックス位置に値を散布（ScatterND）するモデル

    このモデルは指定インデックスの位置に値を配置します。
    演算: インデックスで指定された位置に更新値を書き込む

    用途:
    - スパーステンソルの構築
    - 動的な値の更新
    - インデックスベースの書き込み操作

    注意:
    - Gatherの逆操作に相当
    - 同じインデックスに複数の値がある場合、動作は実装依存
    - scatter_は元のテンソルを変更する破壊的操作
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(ScatterNDModel, self).__init__()

    def forward(self, data, indices, updates):
        """
        順伝播（フォワードパス）の定義

        Args:
            data: ベーステンソル、値を書き込む先のテンソル
            indices: インデックステンソル（整数型）、書き込み位置を指定
            updates: 更新値テンソル、書き込む値を指定

        Returns:
            更新されたテンソル、dataと同じ形状
        """
        # PyTorchのscatter_を使用（破壊的操作）
        # dim=1: 1次元目（列方向）に沿って散布
        return data.scatter_(dim=1, index=indices, src=updates)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = ScatterNDModel()

# ダミーデータの作成（ベーステンソル）
# - torch.zeros: すべて0のテンソルを作成
# - 形状 (2, 5): バッチサイズ2、5要素
dummy_data = torch.zeros(2, 5)

# ダミーインデックスの作成
# - torch.tensor: 指定された値でテンソルを作成
# - .long(): 整数型（int64）に変換
# - 形状 (2, 3): バッチサイズ2、各バッチで3つの位置を指定
dummy_indices = torch.tensor([[0, 1, 2], [2, 3, 4]]).long()

# ダミー更新値の作成
# - torch.tensor: 指定された値でテンソルを作成
# - 形状 (2, 3): indicesと同じ形状、各インデックス位置に書き込む値
dummy_updates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                              # エクスポートするPyTorchモデル
    (dummy_data, dummy_indices, dummy_updates), # 3つの入力をタプルで指定
    "39_scatter_nd.onnx",               # 出力ファイル名（ONNX形式）
    input_names=["data", "indices", "updates"], # ONNX グラフでの入力テンソルの名前
    output_names=["output"],            # ONNX グラフでの出力テンソルの名前
    dynamo=False                        # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 39_scatter_nd.onnx")
