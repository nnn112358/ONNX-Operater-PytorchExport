
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class GatherModel(nn.Module):
    """
    インデックスで要素を収集するモデル
    Gather: 指定インデックスの要素を取得
    """

    def __init__(self):
        super(GatherModel, self).__init__()

    def forward(self, x, indices):
        return torch.gather(x, dim=1, index=indices)


model = GatherModel()
dummy_input = torch.randn(2, 5, 10)
dummy_indices = torch.tensor([[[0, 1, 2], [3, 4, 0], [1, 2, 3], [4, 0, 1], [2, 3, 4]]]).expand(2, 5, 3)
torch.onnx.export(model, (dummy_input, dummy_indices), "38_gather.onnx",
                  input_names=["input", "indices"], output_names=["output"], dynamo=False)
print("saved: 38_gather.onnx")
