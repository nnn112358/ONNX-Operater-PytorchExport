
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class AddModel(nn.Module):
    """
    2つのテンソルの要素ごとの加算を行うモデル
    Add: A + B
    """

    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x, y):
        return x + y


model = AddModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "19_add.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 19_add.onnx")
