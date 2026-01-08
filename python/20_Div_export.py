
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class DivModel(nn.Module):
    """
    2つのテンソルの要素ごとの除算を行うモデル
    Div: A / B
    """

    def __init__(self):
        super(DivModel, self).__init__()

    def forward(self, x, y):
        return x / y


model = DivModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32) + 1.0  # ゼロ除算を避けるため
torch.onnx.export(model, (dummy_input1, dummy_input2), "20_div.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 20_div.onnx")
