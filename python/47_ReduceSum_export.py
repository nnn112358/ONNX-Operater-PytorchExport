
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceSumModel(nn.Module):
    """
    指定軸に沿った合計を計算するモデル
    ReduceSum: 指定次元の要素を合計
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceSumModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


model = ReduceSumModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "47_reduce_sum.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 47_reduce_sum.onnx")
