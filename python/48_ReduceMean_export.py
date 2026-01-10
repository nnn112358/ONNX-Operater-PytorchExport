
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class ReduceMeanModel(nn.Module):
    """
    指定軸に沿った平均を計算するモデル

    このモデルは指定軸に沿って指定軸に沿った平均を計算します。
    演算: Σx / n

    用途:
    - 特徴マップの統計量計算
    - 次元削減
    - 損失関数の計算

    注意:
    - keepdim=Trueの場合、削減後も次元数は保持される（サイズ1の次元として）
    - keepdim=Falseの場合、削減された次元は削除される
    """

    def __init__(self, dim=1, keepdim=True):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、削減パラメータを設定する

        Args:
            dim: 削減する軸のインデックス（デフォルト: 1、チャンネル軸）
            keepdim: 削減後に次元を保持するか（デフォルト: True）
        """
        super(ReduceMeanModel, self).__init__()
        # 削減する軸を保存
        self.dim = dim
        # 削減後に次元を保持するかどうかを保存
        self.keepdim = keepdim

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は任意（例: (batch_size, channels, height, width)）

        Returns:
            削減された結果のテンソル
            例: 入力(1, 3, 32, 32)でdim=1、keepdim=Trueの場合、出力は(1, 1, 32, 32)
        """
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - dim=1: チャンネル軸に沿って削減
# - keepdim=True: 削減後も次元を保持
model = ReduceMeanModel(dim=1, keepdim=True)

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "48_reduce_mean.onnx",       # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 48_reduce_mean.onnx")
