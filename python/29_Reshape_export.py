
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReshapeModel(nn.Module):
    """
    テンソルの形状を変更するモデル
    Reshape: テンソルの要素数を保ったまま形状を変更
    """

    def __init__(self, target_shape):
        super(ReshapeModel, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape(self.target_shape)


model = ReshapeModel(target_shape=(1, -1))
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "29_reshape.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 29_reshape.onnx")
