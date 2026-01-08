
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class HardSwishModel(nn.Module):
    """
    Hard Swish活性化関数を持つニューラルネットワークモデル
    Hard Swish: x * hard_sigmoid(x)
    Swishの高速な近似版です。
    """

    def __init__(self):
        super(HardSwishModel, self).__init__()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        return self.hard_swish(x)


model = HardSwishModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "17_hard_swish.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 17_hard_swish.onnx")
