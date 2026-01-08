
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class MatMulModel(nn.Module):
    """
    行列乗算を行うモデル
    MatMul: 2つの行列の積を計算
    """

    def __init__(self):
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


model = MatMulModel()
dummy_input1 = torch.randn(1, 4, 8)
dummy_input2 = torch.randn(1, 8, 16)
torch.onnx.export(model, (dummy_input1, dummy_input2), "40_matmul.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 40_matmul.onnx")
