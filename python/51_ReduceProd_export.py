
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReduceProdModel(nn.Module):
    """
    指定軸に沿った積を計算するモデル
    ReduceProd: 指定次元の要素の積
    """

    def __init__(self, dim=1, keepdim=True):
        super(ReduceProdModel, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim)


model = ReduceProdModel(dim=1, keepdim=True)
dummy_input = torch.randn(1, 3, 4, 4) * 0.1  # 小さい値で積のオーバーフローを防ぐ
torch.onnx.export(model, dummy_input, "51_reduce_prod.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 51_reduce_prod.onnx")
