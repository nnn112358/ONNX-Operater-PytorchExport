
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class DepthToSpaceModel(nn.Module):
    """
    チャンネル次元を空間次元に変換するモデル
    DepthToSpace: チャンネル数を減らしてHxW空間を増やす
    """

    def __init__(self, block_size=2):
        super(DepthToSpaceModel, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        # PyTorchのpixel_shuffle（DepthToSpaceと同等）
        return torch.nn.functional.pixel_shuffle(x, self.block_size)


model = DepthToSpaceModel(block_size=2)
dummy_input = torch.randn(1, 12, 16, 16)
torch.onnx.export(model, dummy_input, "59_depth_to_space.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 59_depth_to_space.onnx")
