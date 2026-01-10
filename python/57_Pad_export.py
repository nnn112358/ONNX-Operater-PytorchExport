
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn
# 関数モジュールをインポート（pad関数を使用）
import torch.nn.functional as F


class PadModel(nn.Module):
    """
    テンソルにパディングを追加するモデル

    このモデルはテンソルの周囲に値を追加します。
    演算: テンソルの境界に指定値を追加

    用途:
    - 畳み込み層の前処理（サイズ調整）
    - 境界の拡張
    - ゼロパディング

    注意:
    - modeによってパディング方法が異なる（constant, reflect, replicate など）
    - padは(left, right, top, bottom)の順で指定
    """

    def __init__(self, pad=(1, 1, 1, 1), mode='constant', value=0):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、パディングパラメータを設定する

        Args:
            pad: パディングサイズ（デフォルト: (1, 1, 1, 1)）
                 (左, 右, 上, 下)の順で指定
            mode: パディングモード（デフォルト: 'constant'）
            value: 定数パディングの値（デフォルト: 0）
        """
        super(PadModel, self).__init__()
        # パディングサイズを保存
        self.pad = pad
        # パディングモードを保存
        self.mode = mode
        # パディング値を保存
        self.value = value

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            パディングされたテンソル
            例: 入力(1, 3, 32, 32)でpad=(1,1,1,1)の場合、出力は(1, 3, 34, 34)
        """
        return F.pad(x, self.pad, mode=self.mode, value=self.value)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - pad=(1, 1, 1, 1): 左右上下に1ピクセルずつパディング
# - mode='constant': 定数値でパディング
# - value=0: パディング値は0（ゼロパディング）
model = PadModel(pad=(1, 1, 1, 1), mode='constant', value=0)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - パディング後: (1, 3, 34, 34)
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "57_pad.onnx",              # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 57_pad.onnx")
