
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SwishModel(nn.Module):
    """
    Swish活性化関数を持つニューラルネットワークモデル
    Swish: x * sigmoid(x)
    PyTorchではSiLU (Sigmoid Linear Unit)として実装されています。
    """

    def __init__(self):
        super(SwishModel, self).__init__()
        self.swish = nn.SiLU()  # Swish = SiLU

    def forward(self, x):
        return self.swish(x)


model = SwishModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "13_swish.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 13_swish.onnx")
