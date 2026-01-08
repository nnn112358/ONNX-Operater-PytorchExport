
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceMeanModel(nn.Module):
    """
    指定軸に沿った平均を計算するモデル
    ReduceMean: 指定次元の要素の平均
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceMeanModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


model = ReduceMeanModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "48_reduce_mean.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 48_reduce_mean.onnx")
