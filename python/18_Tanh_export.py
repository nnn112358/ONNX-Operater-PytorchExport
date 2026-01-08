
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class TanhModel(nn.Module):
    """
    Tanh活性化関数を持つニューラルネットワークモデル
    Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    出力を-1から1の範囲に変換します。
    """

    def __init__(self):
        super(TanhModel, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


model = TanhModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "18_tanh.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 18_tanh.onnx")
