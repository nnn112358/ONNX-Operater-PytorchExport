
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ClipModel(nn.Module):
    """
    テンソルの値を指定範囲にクリップ（制限）するモデル

    このモデルは入力値を最小値と最大値の範囲内に制限します。
    演算: output[i] = max(min_val, min(input[i], max_val))

    用途:
    - 勾配クリッピング（勾配爆発の防止）
    - 値の範囲を制限（例: 画素値を0-1や0-255の範囲に制限）
    - ReLU6活性化関数の実装（0-6の範囲に制限）

    注意:
    - min_val <= max_val である必要がある
    - 制限された値は勾配が0になる（backpropagation時）
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、クリップ範囲を設定する

        Args:
            min_val: 最小値（デフォルト: -1.0）
            max_val: 最大値（デフォルト: 1.0）
        """
        super(ClipModel, self).__init__()
        # クリップの最小値を設定
        self.min_val = min_val
        # クリップの最大値を設定
        self.max_val = max_val

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            指定範囲にクリップされたテンソル、形状は入力と同じ
        """
        return torch.clamp(x, self.min_val, self.max_val)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - min_val=-1.0: 下限を-1.0に設定
# - max_val=1.0: 上限を1.0に設定
model = ClipModel(min_val=-1.0, max_val=1.0)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
# - 正規分布の値なので-1.0から1.0の範囲外の値も含まれ、クリップ効果を確認できる
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "28_clip.onnx",             # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 28_clip.onnx")
