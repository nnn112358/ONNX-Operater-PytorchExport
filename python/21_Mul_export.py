
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class MulModel(nn.Module):
    """
    2つのテンソルの要素ごとの乗算を行うモデル
    Mul: A * B
    """

    def __init__(self):
        super(MulModel, self).__init__()

    def forward(self, x, y):
        return x * y


model = MulModel()
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "21_mul.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 21_mul.onnx")
