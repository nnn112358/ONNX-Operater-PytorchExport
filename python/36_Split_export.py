
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class SplitModel(nn.Module):
    """
    テンソルを分割するモデル
    Split: 指定軸に沿ってテンソルを分割
    """

    def __init__(self, split_size, dim=1):
        super(SplitModel, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        splits = torch.split(x, self.split_size, dim=self.dim)
        return splits[0], splits[1]


model = SplitModel(split_size=2, dim=1)
dummy_input = torch.randn(1, 4, 32, 32)
torch.onnx.export(model, dummy_input, "36_split.onnx",
                  input_names=["input"], output_names=["output1", "output2"], dynamo=False)
print("saved: 36_split.onnx")
