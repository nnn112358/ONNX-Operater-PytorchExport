
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SpaceToDepthModel(nn.Module):
    """
    空間次元をチャンネル次元に変換するモデル
    SpaceToDepth: HxW空間を減らしてチャンネル数を増やす
    """

    def __init__(self, block_size=2):
        super(SpaceToDepthModel, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        # PyTorchのpixel_unshuffle（SpaceToDepthと同等）
        return torch.nn.functional.pixel_unshuffle(x, self.block_size)


model = SpaceToDepthModel(block_size=2)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "58_space_to_depth.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 58_space_to_depth.onnx")
