
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class EqualModel(nn.Module):
    """
    要素ごとの等価比較を行うモデル
    Equal: A == B
    """

    def __init__(self):
        super(EqualModel, self).__init__()

    def forward(self, x, y):
        return torch.eq(x, y)


model = EqualModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "42_equal.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 42_equal.onnx")
