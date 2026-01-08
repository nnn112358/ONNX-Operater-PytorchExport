
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SigmoidModel(nn.Module):
    """
    Sigmoid活性化関数を持つニューラルネットワークモデル
    Sigmoid: 1 / (1 + exp(-x))
    出力を0から1の範囲に変換します。
    """

    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


model = SigmoidModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "15_sigmoid.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 15_sigmoid.onnx")
