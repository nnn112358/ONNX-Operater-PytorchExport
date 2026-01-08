
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class GreaterModel(nn.Module):
    """
    要素ごとの大なり比較を行うモデル
    Greater: A > B
    """

    def __init__(self):
        super(GreaterModel, self).__init__()

    def forward(self, x, y):
        return torch.gt(x, y)


model = GreaterModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "43_greater.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 43_greater.onnx")
