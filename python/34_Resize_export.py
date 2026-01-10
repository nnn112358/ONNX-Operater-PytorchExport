
# PyTorchの基本ライブラリをインポート
import torch
# ONNXエクスポート機能をインポート
import torch.onnx
# ニューラルネットワークモジュールをインポート
import torch.nn as nn
# 関数モジュールをインポート（interpolate関数を使用）
import torch.nn.functional as F


class ResizeModel(nn.Module):
    """
    テンソルをリサイズ（拡大・縮小）するモデル

    このモデルは補間を使用してテンソルのサイズを変更します。
    演算: 指定スケール係数または目標サイズに応じてリサイズ

    用途:
    - 画像のアップサンプリング（拡大）
    - 画像のダウンサンプリング（縮小）
    - 異なる解像度のレイヤー間の接続

    注意:
    - modeによって補間方法が異なる（nearest, linear, bilinear, bicubic など）
    - align_cornersは補間時のピクセル配置方法を制御
    """

    def __init__(self, scale_factor=2.0, mode='bilinear'):
        """
        モデルの初期化
        親クラス(nn.Module)の初期化を呼び出し、リサイズパラメータを設定する

        Args:
            scale_factor: スケール係数（デフォルト: 2.0、2倍に拡大）
            mode: 補間モード（デフォルト: 'bilinear'、双線形補間）
        """
        super(ResizeModel, self).__init__()
        # スケール係数を保存
        self.scale_factor = scale_factor
        # 補間モードを保存
        self.mode = mode

    def forward(self, x):
        """
        順伝播（フォワードパス）の定義

        Args:
            x: 入力テンソル、形状は (batch_size, channels, height, width)

        Returns:
            リサイズされたテンソル
            例: 入力(1, 3, 16, 16)でscale_factor=2.0の場合、出力は(1, 3, 32, 32)
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


# ==============================================================================
# メイン処理: モデルをONNX形式でエクスポート
# ==============================================================================

# モデルのインスタンスを作成
# - scale_factor=2.0: 2倍に拡大
# - mode='bilinear': 双線形補間を使用
model = ResizeModel(scale_factor=2.0, mode='bilinear')

# ダミー入力の作成
# - torch.randn: 正規分布に従うランダムなテンソルを生成
# - 形状 (1, 3, 16, 16): バッチサイズ1、3チャンネル、16x16ピクセル
# - リサイズ後: (1, 3, 32, 32)
dummy_input = torch.randn(1, 3, 16, 16)

# ONNX形式でモデルをエクスポート
torch.onnx.export(
    model,                      # エクスポートするPyTorchモデル
    dummy_input,                # モデルの入力形状を推論するためのダミー入力
    "34_resize.onnx",           # 出力ファイル名（ONNX形式）
    input_names=["input"],      # ONNX グラフでの入力テンソルの名前
    output_names=["output"],    # ONNX グラフでの出力テンソルの名前
    dynamo=False                # 従来のトレースベースのエクスポートを使用
)

# エクスポート完了メッセージを表示
print("saved: 34_resize.onnx")
