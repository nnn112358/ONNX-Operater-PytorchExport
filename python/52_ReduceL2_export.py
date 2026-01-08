
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceL2Model(nn.Module):
    """
    L2ノルムを計算するモデル
    ReduceL2: √(Σx²)
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceL2Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.norm(x, p=2, dim=self.dim, keepdim=self.keepdim)


model = ReduceL2Model(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "52_reduce_l2.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 52_reduce_l2.onnx")
