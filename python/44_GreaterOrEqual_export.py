
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class GreaterOrEqualModel(nn.Module):
    """
    要素ごとの以上比較を行うモデル
    GreaterOrEqual: A >= B
    """

    def __init__(self):
        super(GreaterOrEqualModel, self).__init__()

    def forward(self, x, y):
        return torch.ge(x, y)


model = GreaterOrEqualModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "44_greater_or_equal.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 44_greater_or_equal.onnx")
