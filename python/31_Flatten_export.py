
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class FlattenModel(nn.Module):
    """
    テンソルを平坦化するモデル
    Flatten: 多次元テンソルを2次元に変換
    """

    def __init__(self, start_dim=1):
        super(FlattenModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=start_dim)

    def forward(self, x):
        return self.flatten(x)


model = FlattenModel(start_dim=1)
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "31_flatten.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 31_flatten.onnx")
