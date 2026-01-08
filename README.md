#ONNX-Operater-PytorchExport


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

> **Note**: `uv run` は自動的に仮想環境を使用するため、activate不要です。

## リファレンス

- [ONNX公式サイト](https://onnx.ai/)
- [ONNXオペレータリファレンス](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [ONNX Runtime](https://onnxruntime.ai/)

## ライセンス

MIT License
