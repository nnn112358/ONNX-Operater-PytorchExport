
# PyTorchの基本ライブラリをインポート
import torch
import torch.onnx
import torch.nn as nn


class ReverseSequenceModel(nn.Module):
    """
    シーケンスを指定長まで反転するモデル
    ReverseSequence: 特定の軸に沿ってシーケンスを反転
    """

    def __init__(self, dim=1):
        super(ReverseSequenceModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        # PyTorchのflip関数を使用してシーケンスを反転
        return torch.flip(x, dims=[self.dim])


model = ReverseSequenceModel(dim=1)
dummy_input = torch.randn(2, 5, 10)
torch.onnx.export(model, dummy_input, "60_reverse_sequence.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
print("saved: 60_reverse_sequence.onnx")
