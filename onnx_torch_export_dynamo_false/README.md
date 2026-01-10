# ONNX Operators Sample Collection

ONNXの主要なオペレータ60種類のPyTorch実装サンプル集です。各オペレータのエクスポートと推論のコード例を提供しています。

## 対応オペレータ

全60種類（ニューラルネットワーク層、活性化関数、数学演算、テンソル操作、線形代数、比較演算、集約・統計演算など）をカバー。

詳細は [onnx_operators.md](onnx_operators.md) を参照。

各オペレータに以下の2ファイルを用意：
- `XX_OperatorName_export.py` - PyTorchモデルをONNX形式にエクスポート
- `XX_OperatorName_inference.py` - ONNXモデルで推論を実行

## セットアップ

### 必要な環境
- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー

### インストール

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# または
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 依存関係のインストール
cd python
uv sync
```

### 依存ライブラリ
- torch (>=2.9.1) - CPU版
- onnx (>=1.16.0)
- onnxruntime (>=1.18.0)

## 使い方

```bash
cd python

# エクスポート（PyTorch → ONNX）
uv run 01_Conv_export.py

# 推論実行
uv run 01_Conv_inference.py
```

他のオペレータも同様です（例：`uv run 14_Softmax_export.py`）

### すべてのオペレータを一括実行

01から60までのすべてのオペレータを順番に実行するスクリプトも用意しています。

```bash
cd python
./run_all.sh
```

このスクリプトは各オペレータのエクスポートと推論を順番に実行し、エラーが発生した場合は処理を停止します。

> **Note**: `uv run` は自動的に仮想環境を使用するため、activate不要です。

## ONNXモデルの可視化

生成されたONNXモデルはNetron.appで可視化できます。以下のリンクをクリックすると、ブラウザ上でモデル構造を確認できます。

### ニューラルネットワーク層
- [01_conv.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/01_conv.onnx)
- [02_conv_transpose.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/02_conv_transpose.onnx)
- [03_maxpool.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/03_maxpool.onnx)
- [04_averagepool.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/04_averagepool.onnx)
- [05_global_averagepool.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/05_global_averagepool.onnx)
- [06_layer_normalization.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/06_layer_normalization.onnx)
- [07_lstm.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/07_lstm.onnx)
- [08_gru.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/08_gru.onnx)

### 活性化関数
- [09_relu.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/09_relu.onnx)
- [10_leaky_relu.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/10_leaky_relu.onnx)
- [11_elu.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/11_elu.onnx)
- [12_prelu.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/12_prelu.onnx)
- [13_swish.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/13_swish.onnx)
- [14_softmax.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/14_softmax.onnx)
- [15_sigmoid.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/15_sigmoid.onnx)
- [16_hard_sigmoid.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/16_hard_sigmoid.onnx)
- [17_hard_swish.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/17_hard_swish.onnx)
- [18_tanh.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/18_tanh.onnx)

### 数学演算
- [19_add.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/19_add.onnx)
- [20_div.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/20_div.onnx)
- [21_mul.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/21_mul.onnx)
- [22_neg.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/22_neg.onnx)
- [23_pow.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/23_pow.onnx)
- [24_sub.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/24_sub.onnx)
- [25_exp.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/25_exp.onnx)
- [26_log.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/26_log.onnx)
- [27_sqrt.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/27_sqrt.onnx)
- [28_clip.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/28_clip.onnx)

### テンソル操作
- [29_reshape.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/29_reshape.onnx)
- [30_transpose.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/30_transpose.onnx)
- [31_flatten.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/31_flatten.onnx)
- [32_squeeze.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/32_squeeze.onnx)
- [33_unsqueeze.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/33_unsqueeze.onnx)
- [34_resize.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/34_resize.onnx)
- [35_concat.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/35_concat.onnx)
- [36_split.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/36_split.onnx)
- [37_slice.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/37_slice.onnx)
- [38_gather.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/38_gather.onnx)
- [39_scatter_nd.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/39_scatter_nd.onnx)

### 線形代数
- [40_matmul.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/40_matmul.onnx)
- [41_gemm.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/41_gemm.onnx)

### 比較演算
- [42_equal.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/42_equal.onnx)
- [43_greater.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/43_greater.onnx)
- [44_greater_or_equal.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/44_greater_or_equal.onnx)
- [45_less.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/45_less.onnx)
- [46_less_or_equal.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/46_less_or_equal.onnx)

### 集約・統計演算
- [47_reduce_sum.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/47_reduce_sum.onnx)
- [48_reduce_mean.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/48_reduce_mean.onnx)
- [49_reduce_max.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/49_reduce_max.onnx)
- [50_reduce_min.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/50_reduce_min.onnx)
- [51_reduce_prod.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/51_reduce_prod.onnx)
- [52_reduce_l2.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/52_reduce_l2.onnx)
- [53_reduce_l1.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/53_reduce_l1.onnx)
- [54_reduce_sum_square.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/54_reduce_sum_square.onnx)
- [55_reduce_log_sum_exp.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/55_reduce_log_sum_exp.onnx)
- [56_reduce_log_sum.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/56_reduce_log_sum.onnx)

### その他の操作
- [57_pad.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/57_pad.onnx)
- [58_space_to_depth.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/58_space_to_depth.onnx)
- [59_depth_to_space.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/59_depth_to_space.onnx)
- [60_reverse_sequence.onnx](https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/ONNX-Operater-PytorchExport/main/onnx_torch_export_dynamo_false/60_reverse_sequence.onnx)

## リファレンス

- [ONNX公式サイト](https://onnx.ai/)
- [ONNXオペレータリファレンス](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [ONNX Runtime](https://onnxruntime.ai/)

## ライセンス

MIT License
