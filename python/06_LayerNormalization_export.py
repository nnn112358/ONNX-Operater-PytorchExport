
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class LayerNormalizationModel(nn.Module):
    """
    レイヤー正規化を持つニューラルネットワークモデル

    レイヤー正規化は、各サンプルの特徴次元にわたって正規化を行います。
    入力: (batch_size, features) の形状のテンソル
    出力: 正規化された同じ形状のテンソル
    """

    def __init__(self, normalized_shape=128):
        """
        モデルの初期化

        Args:
            normalized_shape: 正規化する次元のサイズ
        """
        super(LayerNormalizationModel, self).__init__()

        # レイヤー正規化層の定義
        # - normalized_shape: 正規化する形状（最後の次元）
        # - eps: ゼロ除算を防ぐための小さな値
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=1e-5
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル

        Returns:
            レイヤー正規化を通過した出力テンソル
        """
        return self.layer_norm(x)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = LayerNormalizationModel(normalized_shape=128)

# ダミー入力の作成（バッチサイズ4、128次元の特徴ベクトル）
dummy_input = torch.randn(4, 128)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                              # エクスポートするPyTorchモデル
    dummy_input,                        # モデルの入力形状を推論するためのダミー入力
    "06_layer_normalization.onnx",     # 出力ファイル名（ONNX形式）
    input_names=["input"],              # ONNX グラフでの入力テンソルの名前
    output_names=["output"],            # ONNX グラフでの出力テンソルの名前
    dynamo=False                        # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 06_layer_normalization.onnx")
