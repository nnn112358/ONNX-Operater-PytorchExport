
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class MatMulModel(nn.Module):
    """
    行列乗算（Matrix Multiplication）を行うモデル

    このモデルは2つの行列の積を計算します。
    演算: A @ B (行列の積)

    用途:
    - 線形変換
    - 全結合層の実装
    - アテンション機構のQuery-Key積

    注意:
    - バッチ演算をサポート（バッチ次元は保持される）
    - 入力行列の内側の次元は一致する必要がある（A: m×n、B: n×p）
    - 出力形状は外側の次元の組み合わせ（結果: m×p）
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1の入力テンソル（行列）、形状は (batch_size, m, n)
            y: 第2の入力テンソル（行列）、形状は (batch_size, n, p)

        Returns:
            行列積のテンソル、形状は (batch_size, m, p)
            例: (1, 4, 8) @ (1, 8, 16) = (1, 4, 16)
        """
        return torch.matmul(x, y)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = MatMulModel()

# ダミー入力1の作成（第1の行列）
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 4, 8): バッチサイズ1、4行、8列
dummy_input1 = torch.randn(1, 4, 8)

# ダミー入力2の作成（第2の行列）
# - 形状 (1, 8, 16): バッチサイズ1、8行、16列
# - 入力1の列数（8）と入力2の行数（8）が一致している必要がある
dummy_input2 = torch.randn(1, 8, 16)

# ONNX形式でモデルをエクスポート
# 出力形状: (1, 4, 16) - バッチサイズ1、4行、16列
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),   # 複数の入力をタプルで指定
    "40_matmul.onnx",               # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"], # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 40_matmul.onnx")
