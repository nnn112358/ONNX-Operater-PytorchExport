
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class GemmModel(nn.Module):
    """
    一般行列乗算（General Matrix Multiplication）を行うモデル

    このモデルは一般的な行列演算を実行します。
    演算: Y = alpha * A * B + beta * C
    （PyTorchのLinear層は内部的にGemmとしてONNXにエクスポートされます）

    用途:
    - 全結合層（Fully Connected Layer）
    - アフィン変換
    - 線形回帰

    注意:
    - Linear層は重み行列とバイアスを含む
    - ONNXでは最適化されたGemm演算として表現される
    - バイアスはオプショナル
    """

    def __init__(self, in_features=8, out_features=16):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、全結合層を定義する

        Args:
            in_features: 入力特徴量の次元数（デフォルト: 8）
            out_features: 出力特徴量の次元数（デフォルト: 16）
        """
        super(GemmModel, self).__init__()
        # 全結合層を定義
        # - in_features=8: 入力次元数
        # - out_features=16: 出力次元数
        # - bias=True (デフォルト): バイアス項を含む
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, in_features)
               例: (4, 8) = バッチサイズ4、入力次元8

        Returns:
            線形変換された出力テンソル、形状は (batch_size, out_features)
            例: (4, 16) = バッチサイズ4、出力次元16
        """
        return self.linear(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - in_features=8: 入力特徴量8次元
# - out_features=16: 出力特徴量16次元
model = GemmModel(in_features=8, out_features=16)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (4, 8): バッチサイズ4、入力次元8
# - 出力形状: (4, 16)
dummy_input = torch.randn(4, 8)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "41_gemm.onnx",             # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 41_gemm.onnx")
