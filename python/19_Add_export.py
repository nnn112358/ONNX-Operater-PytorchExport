
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class AddModel(nn.Module):
    """
    2つのテンソルの要素ごとの加算を行うモデル

    Add: A + B
    ブロードキャスト機能により、異なる形状のテンソル同士の加算も可能です。
    ResNetなどでスキップ接続を実装する際によく使用されます。
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す
        """
        super(AddModel, self).__init__()

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1入力テンソル
            y: 第2入力テンソル

        Returns:
            2つのテンソルの要素ごとの和
        """
        return x + y


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = AddModel()

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32):
#   - 1  : バッチサイズ（1つのサンプル）
#   - 3  : チャンネル数（RGB画像を想定）
#   - 32 : 画像の高さ（ピクセル）
#   - 32 : 画像の幅（ピクセル）
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
# ONNX (Open Neural Network Exchange) は異なるフレームワーク間で
# モデルを共有するための標準フォーマット
torch.onnx.export(
    model,                              # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),       # モデルの入力形状を推論するためのダミー入力（タプル）
    "19_add.onnx",                      # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"],   # ONNX グラフでの入力テンソルの名前
    output_names=["output"],            # ONNX グラフでの出力テンソルの名前
    dynamo=False                        # 従来のトレースベースのエクスポートを使用
                                        # （TorchDynamoベースのエクスポートは使用しない）
)

# エクスポート完了メッセージを表示
print("saved: 19_add.onnx")
