
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceLogSumModel(nn.Module):
    """
    log(Σx)を計算するモデル
    ReduceLogSum: log(Σx)
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceLogSumModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.log(torch.sum(x, dim=self.dim, keepdim=self.keepdim))


model = ReduceLogSumModel(dim=1, keepdim=True)
dummy_input = torch.abs(torch.randn(1, 3, 32, 32))  # 正の値
torch.onnx.export(model, dummy_input, "56_reduce_log_sum.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 56_reduce_log_sum.onnx")
