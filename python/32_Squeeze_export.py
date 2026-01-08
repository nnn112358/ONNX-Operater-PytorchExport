
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SqueezeModel(nn.Module):
    """
    サイズ1の次元を削除するモデル
    Squeeze: 形状から1の次元を除去
    """

    def __init__(self, dim=None):
        super(SqueezeModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is not None:
            return x.squeeze(self.dim)
        return x.squeeze()


model = SqueezeModel(dim=1)
dummy_input = torch.randn(1, 1, 32, 32)
torch.onnx.export(model, dummy_input, "32_squeeze.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 32_squeeze.onnx")
