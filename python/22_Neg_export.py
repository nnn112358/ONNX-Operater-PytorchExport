
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class NegModel(nn.Module):
    """
    テンソルの符号を反転（Negation）するモデル

    このモデルは入力テンソルの各要素の符号を反転します。
    演算: output[i] = -input[i]

    用途:
    - 勾配の符号反転
    - 損失関数の最大化問題を最小化問題に変換
    - 対称性を持つ演算の実装

    注意: 整数型、浮動小数点型どちらでも使用可能
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(NegModel, self).__init__()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            符号反転されたテンソル、形状は入力と同じ
        """
        return -x


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = NegModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 正負の値が混在する入力で符号反転をテスト
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "22_neg.onnx",              # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 22_neg.onnx")
