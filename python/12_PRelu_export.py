
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class PReluModel(nn.Module):
    """
    PReLU活性化関数を持つニューラルネットワークモデル
    PReLU: x if x > 0 else learnable_parameter * x
    負の傾きが学習可能なパラメータです。
    """

    def __init__(self, num_parameters=1):
        super(PReluModel, self).__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x):
        return self.prelu(x)


model = PReluModel(num_parameters=1)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "12_prelu.onnx", input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 12_prelu.onnx")
