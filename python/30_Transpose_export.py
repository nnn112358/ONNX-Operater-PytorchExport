
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class TransposeModel(nn.Module):
    """
    テンソルの次元を入れ替えるモデル
    Transpose: 次元の順序を変更
    """

    def __init__(self, dim0, dim1):
        super(TransposeModel, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


model = TransposeModel(dim0=2, dim1=3)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "30_transpose.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 30_transpose.onnx")
