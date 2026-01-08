
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class PowModel(nn.Module):
    """
    テンソルのべき乗演算を行うモデル
    Pow: A ^ B
    """

    def __init__(self):
        super(PowModel, self).__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


model = PowModel()
dummy_input1 = torch.abs(torch.randn(1, 3, 32, 32))  # 正の値
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "23_pow.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 23_pow.onnx")
