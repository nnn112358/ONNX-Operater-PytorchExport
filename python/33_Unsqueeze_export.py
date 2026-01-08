
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class UnsqueezeModel(nn.Module):
    """
    サイズ1の次元を追加するモデル
    Unsqueeze: 指定位置に次元を追加
    """

    def __init__(self, dim):
        super(UnsqueezeModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


model = UnsqueezeModel(dim=1)
dummy_input = torch.randn(1, 32, 32)
torch.onnx.export(model, dummy_input, "33_unsqueeze.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 33_unsqueeze.onnx")
