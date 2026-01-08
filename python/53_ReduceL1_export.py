
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceL1Model(nn.Module):
    """
    L1ノルムを計算するモデル
    ReduceL1: Σ|x|
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceL1Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.norm(x, p=1, dim=self.dim, keepdim=self.keepdim)


model = ReduceL1Model(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "53_reduce_l1.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 53_reduce_l1.onnx")
