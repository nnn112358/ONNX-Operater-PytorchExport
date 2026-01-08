
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceSumSquareModel(nn.Module):
    """
    二乗和を計算するモデル
    ReduceSumSquare: Σx²
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceSumSquareModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(x ** 2, dim=self.dim, keepdim=self.keepdim)


model = ReduceSumSquareModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "54_reduce_sum_square.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 54_reduce_sum_square.onnx")
