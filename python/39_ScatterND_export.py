
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ScatterNDModel(nn.Module):
    """
    インデックス位置に値を散布するモデル
    ScatterND: 指定位置に値を配置
    """

    def __init__(self):
        super(ScatterNDModel, self).__init__()

    def forward(self, data, indices, updates):
        # PyTorchのscatter_を使用
        return data.scatter_(dim=1, index=indices, src=updates)


model = ScatterNDModel()
dummy_data = torch.zeros(2, 5)
dummy_indices = torch.tensor([[0, 1, 2], [2, 3, 4]]).long()
dummy_updates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
torch.onnx.export(model, (dummy_data, dummy_indices, dummy_updates), "39_scatter_nd.onnx",
                  input_names=["data", "indices", "updates"], output_names=["output"], dynamo=False)
print("saved: 39_scatter_nd.onnx")
