
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SqrtModel(nn.Module):
    """
    テンソルの平方根（Square Root）を計算するモデル

    このモデルは入力値の平方根を計算します。
    演算: output[i] = √input[i]

    用途:
    - L2ノルムの計算
    - ユークリッド距離の算出
    - 標準偏差の計算

    注意:
    - 入力は非負の値である必要がある（負の値では実数の平方根が定義されない）
    - 入力が0の場合、出力も0となる
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(SqrtModel, self).__init__()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル（非負の値）、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            平方根が適用されたテンソル、形状は入力と同じ
        """
        return torch.sqrt(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = SqrtModel()

# ダミー入力の作成
# - torch.abs: 絶対値を取得して非負の値を保証
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input = torch.abs(torch.randn(1, 3, 32, 32))

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "27_sqrt.onnx",             # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 27_sqrt.onnx")
