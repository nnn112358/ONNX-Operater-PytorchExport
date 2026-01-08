
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class LessModel(nn.Module):
    """
    要素ごとの小なり比較を行うモデル
    Less: A < B
    """

    def __init__(self):
        super(LessModel, self).__init__()

    def forward(self, x, y):
        return torch.lt(x, y)


model = LessModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "45_less.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 45_less.onnx")
