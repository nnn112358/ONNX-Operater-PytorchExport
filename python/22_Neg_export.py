
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class NegModel(nn.Module):
    """
    テンソルの符号を反転するモデル
    Neg: -A
    """

    def __init__(self):
        super(NegModel, self).__init__()

    def forward(self, x):
        return -x


model = NegModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "22_neg.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 22_neg.onnx")
