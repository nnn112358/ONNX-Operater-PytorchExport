
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class GemmModel(nn.Module):
    """
    一般行列乗算を行うモデル
    Gemm: alpha*A*B + beta*C
    PyTorchのLinear層はGemmとしてエクスポートされます
    """

    def __init__(self, in_features=8, out_features=16):
        super(GemmModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


model = GemmModel(in_features=8, out_features=16)
dummy_input = torch.randn(4, 8)
torch.onnx.export(model, dummy_input, "41_gemm.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 41_gemm.onnx")
