
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceLogSumExpModel(nn.Module):
    """
    log(Σe^x)を計算するモデル（数値安定版）
    ReduceLogSumExp: log(Σe^x)
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceLogSumExpModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.logsumexp(x, dim=self.dim, keepdim=self.keepdim)


model = ReduceLogSumExpModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "55_reduce_log_sum_exp.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 55_reduce_log_sum_exp.onnx")
