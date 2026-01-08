
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SqrtModel(nn.Module):
    """
    テンソルの平方根を計算するモデル
    Sqrt: √x
    """

    def __init__(self):
        super(SqrtModel, self).__init__()

    def forward(self, x):
        return torch.sqrt(x)


model = SqrtModel()
dummy_input = torch.abs(torch.randn(1, 3, 32, 32))  # 正の値
torch.onnx.export(model, dummy_input, "27_sqrt.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 27_sqrt.onnx")
