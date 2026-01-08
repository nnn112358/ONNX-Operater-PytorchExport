
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM（Long Short-Term Memory）層を持つニューラルネットワークモデル

    LSTMは時系列データやシーケンスデータを処理するためのRNNの一種です。
    長期的な依存関係を学習できるのが特徴です。
    """

    def __init__(self, input_size=10, hidden_size=20, num_layers=1):
        """
        モデルの初期化

        Args:
            input_size: 入力特徴量の次元数
            hidden_size: 隠れ状態の次元数
            num_layers: LSTMレイヤーの数
        """
        super(LSTMModel, self).__init__()

        # LSTM層の定義
        # - input_size: 入力の特徴次元数
        # - hidden_size: 隠れ状態の次元数
        # - num_layers: LSTMレイヤーの数
        # - batch_first: Trueの場合、入力形状は(batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, sequence_length, input_size)

        Returns:
            output: LSTM層の出力、形状は (batch_size, sequence_length, hidden_size)
        """
        output, (hidden, cell) = self.lstm(x)
        return output


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
model = LSTMModel(input_size=10, hidden_size=20, num_layers=1)

# ダミー入力の作成（バッチサイズ4、シーケンス長5、特徴次元10）
dummy_input = torch.randn(4, 5, 10)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                          # エクスポートするPyTorchモデル
    dummy_input,                    # モデルの入力形状を推論するためのダミー入力
    "07_lstm.onnx",                 # 出力ファイル名（ONNX形式）
    input_names=["input"],          # ONNX グラフでの入力テンソルの名前
    output_names=["output"],        # ONNX グラフでの出力テンソルの名前
    dynamo=False                    # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 07_lstm.onnx")
