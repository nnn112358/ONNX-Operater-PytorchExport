
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SubModel(nn.Module):
    """
    2つのテンソルの要素ごとの減算を行うモデル
    Sub: A - B
    """

    def __init__(self):
        super(SubModel, self).__init__()

    def forward(self, x, y):
        return x - y


model = SubModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "24_sub.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 24_sub.onnx")
