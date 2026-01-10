
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ReshapeModel(nn.Module):
    """
    テンソルの形状を変更（Reshape）するモデル

    このモデルはテンソルの要素数を保ったまま形状を変更します。
    演算: テンソルの次元構造を再構成

    用途:
    - 全結合層への入力準備（4D→2Dへの変換など）
    - バッチ処理のための形状変更
    - モデル間のテンソル形状の調整

    注意:
    - 元の要素数と新しい形状の要素数は一致する必要がある
    - -1を指定すると、その次元のサイズは自動的に計算される
    """

    def __init__(self, target_shape):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、目標形状を設定する

        Args:
            target_shape: 変換後のテンソル形状（例: (1, -1)）
                         -1は自動計算を意味する
        """
        super(ReshapeModel, self).__init__()
        # 変換後の目標形状を保存
        self.target_shape = target_shape

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            形状変更されたテンソル、要素数は入力と同じ
        """
        return x.reshape(self.target_shape)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - target_shape=(1, -1): バッチ次元を保持し、残りを1次元に平坦化
# - 入力形状 (1, 3, 32, 32) の場合、出力形状は (1, 3072) になる
model = ReshapeModel(target_shape=(1, -1))

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 総要素数: 1 * 3 * 32 * 32 = 3072
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "29_reshape.onnx",          # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 29_reshape.onnx")
