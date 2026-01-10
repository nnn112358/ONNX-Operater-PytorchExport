
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class HardSigmoidModel(nn.Module):
    """
    Hard Sigmoid活性化関数を持つニューラルネットワークモデル

    Hard Sigmoid: Sigmoidの区分線形近似版で計算が高速です。
    ReLU6ベースの近似により、指数関数を使わずに実装できるため、
    モバイルデバイスなどの計算リソースが限られた環境で有効です。
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、Hard Sigmoid活性化関数を定義する
        """
        super(HardSigmoidModel, self).__init__()
        # Hard Sigmoid活性化関数の定義
        # Sigmoidの高速な区分線形近似版
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            Hard Sigmoid活性化関数を適用した出力テンソル
        """
        return self.hard_sigmoid(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = HardSigmoidModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32):
#   - 1  : バッチサイズ（1つのサンプル）
#   - 3  : チャンネル数（RGB画像を想定）
#   - 32 : 画像の高さ（ピクセル）
#   - 32 : 画像の幅（ピクセル）
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
# ONNX (Open Neural Network Exchange) は異なるフレームワーク間で
# モデルを共有するための標準フォーマット
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "16_hard_sigmoid.onnx",     # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
                                # （TorchDynamoベースのエクスポートは使用しない）
)

# エクスポート完了メッセージを表示
print("saved: 16_hard_sigmoid.onnx")
