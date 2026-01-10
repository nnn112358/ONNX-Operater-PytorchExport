
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class PowModel(nn.Module):
    """
    テンソルのべき乗演算（Power）を行うモデル

    このモデルは第1の入力を第2の入力でべき乗します。
    演算: output[i] = input1[i] ^ input2[i]

    用途:
    - 特徴量のべき乗変換
    - 正規化処理（p-norm計算など）
    - 非線形変換の適用

    注意:
    - 底（input1）が負の値で指数（input2）が非整数の場合、結果は複素数になる
    - 数値安定性のため、通常は正の値を底として使用
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(PowModel, self).__init__()

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1の入力テンソル（底）、形状は任意（例: (batch_size, channels, height, width)）
            y: 第2の入力テンソル（指数）、xと同じ形状またはブロードキャスト可能な形状

        Returns:
            べき乗演算の結果のテンソル、形状はxと同じ
        """
        return torch.pow(x, y)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = PowModel()

# ダミー入力1の作成（底）
# - torch.abs: 絶対値を取得して正の値を保証
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 正の値のみを使用することで、任意の指数でも安全に計算可能
dummy_input1 = torch.abs(torch.randn(1, 3, 32, 32))

# ダミー入力2の作成（指数）
# - 形状 (1, 3, 32, 32): 入力1と同じ形状
# - 任意の実数値を指数として使用可能
dummy_input2 = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),   # 複数の入力をタプルで指定
    "23_pow.onnx",                  # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"], # ONNX グラフでの入力テンソルの名前（底、指数）
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 23_pow.onnx")
