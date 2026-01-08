
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceMaxModel(nn.Module):
    """
    指定軸に沿った最大値を計算するモデル
    ReduceMax: 指定次元の最大値
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceMaxModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.max(x, dim=self.dim, keepdim=self.keepdim)[0]


model = ReduceMaxModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "49_reduce_max.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 49_reduce_max.onnx")
