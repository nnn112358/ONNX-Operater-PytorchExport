
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ClipModel(nn.Module):
    """
    テンソルの値を指定範囲にクリップするモデル
    Clip: min <= x <= max
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super(ClipModel, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


model = ClipModel(min_val=-1.0, max_val=1.0)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "28_clip.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 28_clip.onnx")
