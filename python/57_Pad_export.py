
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F


class PadModel(nn.Module):
    """
    テンソルにパディングを追加するモデル
    Pad: テンソルの周囲に値を追加
    """

    def __init__(self, pad=(1, 1, 1, 1), mode='constant', value=0):
        super(PadModel, self).__init__()
        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, x):
        return F.pad(x, self.pad, mode=self.mode, value=self.value)


model = PadModel(pad=(1, 1, 1, 1), mode='constant', value=0)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "57_pad.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 57_pad.onnx")
