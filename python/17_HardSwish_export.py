
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class HardSwishModel(nn.Module):
    """
    Hard Swish活性化関数を持つニューラルネットワークモデル

    Hard Swish: x * hard_sigmoid(x)
    Swishの高速な近似版です。
    MobileNetV3などの軽量モデルで使用され、計算コストを抑えながら
    Swishに近い性能を実現します。
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、Hard Swish活性化関数を定義する
        """
        super(HardSwishModel, self).__init__()
        # Hard Swish活性化関数の定義
        # Swishの高速な区分線形近似版
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            Hard Swish活性化関数を適用した出力テンソル
        """
        return self.hard_swish(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = HardSwishModel()

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
    "17_hard_swish.onnx",       # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
                                # （TorchDynamoベースのエクスポートは使用しない）
)

# エクスポート完了メッセージを表示
print("saved: 17_hard_swish.onnx")
