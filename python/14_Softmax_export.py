
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SoftmaxModel(nn.Module):
    """
    Softmax活性化関数を持つニューラルネットワークモデル
    Softmax: 指定した軸に沿って確率分布に変換します。
    """

    def __init__(self, dim=1):
        super(SoftmaxModel, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


model = SoftmaxModel(dim=1)
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "14_softmax.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 14_softmax.onnx")
