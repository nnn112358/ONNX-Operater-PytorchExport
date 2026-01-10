
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class GatherModel(nn.Module):
    """
    インデックスで要素を収集（Gather）するモデル

    このモデルは指定軸に沿って、インデックスで指定された要素を収集します。
    演算: 指定インデックスの要素を取得

    用途:
    - 埋め込みベクトルの取得
    - インデックスベースの要素選択
    - 分類タスクでの予測値抽出

    注意:
    - インデックスは整数テンソルで指定
    - 指定軸に沿って収集が行われる
    - インデックスの範囲は指定軸のサイズ内である必要がある
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(GatherModel, self).__init__()

    def forward(self, x, indices):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, num_items, feature_dim)）
            indices: インデックステンソル（整数型）、収集する要素のインデックスを指定

        Returns:
            収集された要素のテンソル、形状はindicesの形状に依存
        """
        return torch.gather(x, dim=1, index=indices)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = GatherModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (2, 5, 10): バッチサイズ2、5アイテム、10次元特徴
dummy_input = torch.randn(2, 5, 10)

# ダミーインデックスの作成
# - torch.tensor: 指定された値でテンソルを作成
# - expand: テンソルを指定形状に拡張
# - 形状 (2, 5, 3): バッチサイズ2、5アイテム、各アイテムから3要素を収集
# - dim=1に沿って収集するため、インデックスは0-9の範囲
dummy_indices = torch.tensor([[[0, 1, 2], [3, 4, 0], [1, 2, 3], [4, 0, 1], [2, 3, 4]]]).expand(2, 5, 3)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input, dummy_indices),   # 複数の入力をタプルで指定
    "38_gather.onnx",               # 出力ファイル名（ONNX形式）
    input_names=["input", "indices"], # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 38_gather.onnx")
