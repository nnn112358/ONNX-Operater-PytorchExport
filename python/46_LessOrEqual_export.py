
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class LessOrEqualModel(nn.Module):
    """
    要素ごとの以下比較を行うモデル
    LessOrEqual: A <= B
    """

    def __init__(self):
        super(LessOrEqualModel, self).__init__()

    def forward(self, x, y):
        return torch.le(x, y)


model = LessOrEqualModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "46_less_or_equal.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 46_less_or_equal.onnx")
