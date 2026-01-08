
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceMinModel(nn.Module):
    """
    指定軸に沿った最小値を計算するモデル
    ReduceMin: 指定次元の最小値
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceMinModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.min(x, dim=self.dim, keepdim=self.keepdim)[0]


model = ReduceMinModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "50_reduce_min.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 50_reduce_min.onnx")
