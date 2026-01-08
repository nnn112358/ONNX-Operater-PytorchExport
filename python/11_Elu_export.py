
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class EluModel(nn.Module):
    """
    ELU活性化関数を持つニューラルネットワークモデル
    ELU: x if x > 0 else alpha * (exp(x) - 1)
    """

    def __init__(self, alpha=1.0):
        super(EluModel, self).__init__()
        self.elu = nn.ELU(alpha=alpha)

    def forward(self, x):
        return self.elu(x)


model = EluModel(alpha=1.0)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "11_elu.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 11_elu.onnx")
