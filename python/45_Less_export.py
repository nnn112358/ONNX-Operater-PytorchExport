
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class LessModel(nn.Module):
    """
    要素ごとの小なり比較を行うモデル

    このモデルは2つの入力テンソルの各要素を比較します。
    演算: A < B

    用途:
    - 条件分岐の実装
    - マスク生成
    - 論理演算の構築

    注意:
    - 出力はブール型テンソル（True/False）
    - ブロードキャスト規則が適用される
    """

    def __init__(self):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出す

        このモデルは学習可能なパラメータを持たず、純粋な演算のみを行います
        """
        super(LessModel, self).__init__()

    def forward(self, x, y):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 第1の入力テンソル、形状は任意（例: (batch_size, channels, height, width)）
            y: 第2の入力テンソル、xと同じ形状またはブロードキャスト可能な形状

        Returns:
            比較結果のブール型テンソル、形状はxと同じ
        """
        return torch.lt(x, y)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = LessModel()

# ダミー入力1の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 32, 32): バッチサイズ1、3チャンネル、32x32ピクセル
dummy_input1 = torch.randn(1, 3, 32, 32)

# ダミー入力2の作成
# - 入力1と同じ形状のテンソルを生成（要素ごとの比較のため）
dummy_input2 = torch.randn(1, 3, 32, 32)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    (dummy_input1, dummy_input2),   # 複数の入力をタプルで指定
    "45_less.onnx",           # 出力ファイル名（ONNX形式）
    input_names=["input1", "input2"], # ONNX グラフでの入力テンソルの名前（2つの入力）
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 45_less.onnx")
