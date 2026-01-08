
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ConcatModel(nn.Module):
    """
    複数のテンソルを連結するモデル
    Concat: 指定軸に沿ってテンソルを結合
    """

    def __init__(self, dim=1):
        super(ConcatModel, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat([x, y], dim=self.dim)


model = ConcatModel(dim=1)
dummy_input1 = torch.randn(1, 3, 32, 32)
dummy_input2 = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (dummy_input1, dummy_input2), "35_concat.onnx",
                  input_names=["input1", "input2"], output_names=["output"], dynamo=False)
print("saved: 35_concat.onnx")
