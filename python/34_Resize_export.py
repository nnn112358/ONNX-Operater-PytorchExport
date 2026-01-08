
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F


class ResizeModel(nn.Module):
    """
    テンソルをリサイズ（拡大・縮小）するモデル
    Resize: 補間を使用してテンソルのサイズを変更
    """

    def __init__(self, scale_factor=2.0, mode='bilinear'):
        super(ResizeModel, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


model = ResizeModel(scale_factor=2.0, mode='bilinear')
dummy_input = torch.randn(1, 3, 16, 16)
torch.onnx.export(model, dummy_input, "34_resize.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 34_resize.onnx")
