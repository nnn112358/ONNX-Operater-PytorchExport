
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SliceModel(nn.Module):
    """
    テンソルの一部を切り出すモデル
    Slice: 指定範囲のテンソルを抽出
    """

    def __init__(self):
        super(SliceModel, self).__init__()

    def forward(self, x):
        # チャンネル次元で1:3を切り出し
        return x[:, 1:3, :, :]


model = SliceModel()
dummy_input = torch.randn(1, 4, 32, 32)
torch.onnx.export(model, dummy_input, "37_slice.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 37_slice.onnx")
