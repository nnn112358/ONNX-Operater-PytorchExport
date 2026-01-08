
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ExpModel(nn.Module):
    """
    テンソルの指数関数を計算するモデル
    Exp: e^x
    """

    def __init__(self):
        super(ExpModel, self).__init__()

    def forward(self, x):
        return torch.exp(x)


model = ExpModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "25_exp.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 25_exp.onnx")
