
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class HardSigmoidModel(nn.Module):
    """
    Hard Sigmoid活性化関数を持つニューラルネットワークモデル
    Hard Sigmoid: Sigmoidの区分線形近似版で計算が高速です。
    """

    def __init__(self):
        super(HardSigmoidModel, self).__init__()
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        return self.hard_sigmoid(x)


model = HardSigmoidModel()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "16_hard_sigmoid.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 16_hard_sigmoid.onnx")
