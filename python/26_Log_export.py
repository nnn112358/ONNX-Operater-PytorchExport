
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class LogModel(nn.Module):
    """
    テンソルの自然対数（Natural Logarithm）を計算するモデル

    このモデルは入力値の自然対数（底がe）を計算します。
    演算: output[i] = ln(input[i]) = log_e(input[i])

    用途:
    - 対数尤度の計算
    - 指数スケールのデータを線形スケールに変換
    - 情報理論における エントロピー計算

    注意:
    - 入力は正の値である必要がある（0以下の値では定義されない）
    - 入力が0に近いと結果が-∞に発散する可能性がある
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(LogModel, self).__init__()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル（正の値）、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            自然対数が適用されたテンソル、形状は入力と同じ
        """
        return torch.log(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = LogModel()

# ダミー入力の作成
# - torch.abs: 絶対値を取得して正の値を保証
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - + 0.1: 0に近い値を避けて数値安定性を確保
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input = torch.abs(torch.randn(1, 3, 32, 32)) + 0.1

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "26_log.onnx",              # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 26_log.onnx")
