
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class SubModel(nn.Module):
    """
    2つのテンソルの要素ごとの減算（Subtraction）を行うモデル

    このモデルは第1の入力から第2の入力を引き算します。
    演算: output[i] = input1[i] - input2[i]

    用途:
    - 残差計算（ResNetなどで使用）
    - 画像の差分検出
    - 特徴量の相対的な変化の計算

    注意: ブロードキャスト規則が適用されます（形状が異なる場合）
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(SubModel, self).__init__()

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1の入力テンソル（被減数）、形状は任意（例: (batch_size, channels, height, width)）
            y: 第2の入力テンソル（減数）、xと同じ形状またはブロードキャスト可能な形状

        Returns:
            要素ごとの減算結果のテンソル、形状はxと同じ
        """
        return x - y


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = SubModel()

# ダミー入力1の作成（被減数）
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input1 = torch.randn(1, 3, 32, 32)

# ダミー入力2の作成（減数）
# - 入力1と同じ形状のテンソルを生成（要素ごとの減算のため）
dummy_input2 = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),   # 複数の入力をタプルで指定
    "24_sub.onnx",                  # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"], # ONNX グラフでの入力テンソルの名前（2つの入力）
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 24_sub.onnx")
