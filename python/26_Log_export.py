
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class LogModel(nn.Module):
    """
    テンソルの自然対数を計算するモデル
    Log: ln(x)
    """

    def __init__(self):
        super(LogModel, self).__init__()

    def forward(self, x):
        return torch.log(x)


model = LogModel()
dummy_input = torch.abs(torch.randn(1, 3, 32, 32)) + 0.1  # 正の値
torch.onnx.export(model, dummy_input, "26_log.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 26_log.onnx")
