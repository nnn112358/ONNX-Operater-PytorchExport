# ONNX オペレータ一覧

## 1. ニューラルネットワーク層 (Neural Network Layers)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Conv | 畳み込み演算 | [export](python/01_Conv_export.py) / [inference](python/01_Conv_inference.py) |
| ConvTranspose | 転置畳み込み（逆畳み込み） | [export](python/02_ConvTranspose_export.py) / [inference](python/02_ConvTranspose_inference.py) |
| MaxPool | 最大値プーリング | [export](python/03_MaxPool_export.py) / [inference](python/03_MaxPool_inference.py) |
| AveragePool | 平均値プーリング | [export](python/04_AveragePool_export.py) / [inference](python/04_AveragePool_inference.py) |
| GlobalAveragePool | グローバル平均プーリング | [export](python/05_GlobalAveragePool_export.py) / [inference](python/05_GlobalAveragePool_inference.py) |
| LayerNormalization | レイヤー正規化 | [export](python/06_LayerNormalization_export.py) / [inference](python/06_LayerNormalization_inference.py) |
| LSTM | 長短期記憶ネットワーク | [export](python/07_LSTM_export.py) / [inference](python/07_LSTM_inference.py) |
| GRU | ゲート付き回帰ユニット | [export](python/08_GRU_export.py) / [inference](python/08_GRU_inference.py) |

## 2. 活性化関数 (Activation Functions)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Relu | ReLU活性化関数 (max(0, x)) | [export](python/09_Relu_export.py) / [inference](python/09_Relu_inference.py) |
| LeakyRelu | Leaky ReLU活性化関数 | [export](python/10_LeakyRelu_export.py) / [inference](python/10_LeakyRelu_inference.py) |
| Elu | ELU活性化関数 | [export](python/11_Elu_export.py) / [inference](python/11_Elu_inference.py) |
| PRelu | Parametric ReLU活性化関数 | [export](python/12_PRelu_export.py) / [inference](python/12_PRelu_inference.py) |
| Swish | Swish活性化関数 (x * sigmoid(x)) | [export](python/13_Swish_export.py) / [inference](python/13_Swish_inference.py) |
| Softmax | Softmax関数（確率分布化） | [export](python/14_Softmax_export.py) / [inference](python/14_Softmax_inference.py) |
| Sigmoid | シグモイド関数 (1/(1+e^-x)) | [export](python/15_Sigmoid_export.py) / [inference](python/15_Sigmoid_inference.py) |
| HardSigmoid | Hard Sigmoid（区分線形近似） | [export](python/16_HardSigmoid_export.py) / [inference](python/16_HardSigmoid_inference.py) |
| HardSwish | Hard Swish活性化関数 | [export](python/17_HardSwish_export.py) / [inference](python/17_HardSwish_inference.py) |
| Tanh | ハイパボリックタンジェント | [export](python/18_Tanh_export.py) / [inference](python/18_Tanh_inference.py) |

## 3. 数学演算 (Math Operations)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Add | 2つのテンソルの要素ごとの加算 | [export](python/19_Add_export.py) / [inference](python/19_Add_inference.py) |
| Div | 2つのテンソルの要素ごとの除算 | [export](python/20_Div_export.py) / [inference](python/20_Div_inference.py) |
| Mul | 2つのテンソルの要素ごとの乗算 | [export](python/21_Mul_export.py) / [inference](python/21_Mul_inference.py) |
| Neg | テンソルの符号を反転 | [export](python/22_Neg_export.py) / [inference](python/22_Neg_inference.py) |
| Pow | テンソルのべき乗演算 | [export](python/23_Pow_export.py) / [inference](python/23_Pow_inference.py) |
| Sub | 2つのテンソルの要素ごとの減算 | [export](python/24_Sub_export.py) / [inference](python/24_Sub_inference.py) |
| Exp | テンソルの指数関数 (e^x) | [export](python/25_Exp_export.py) / [inference](python/25_Exp_inference.py) |
| Log | テンソルの自然対数 | [export](python/26_Log_export.py) / [inference](python/26_Log_inference.py) |
| Sqrt | テンソルの平方根 | [export](python/27_Sqrt_export.py) / [inference](python/27_Sqrt_inference.py) |
| Clip | テンソルの値を指定範囲にクリップ | [export](python/28_Clip_export.py) / [inference](python/28_Clip_inference.py) |

## 4. テンソル操作 (Tensor Operations)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Reshape | テンソルの形状を変更 | [export](python/29_Reshape_export.py) / [inference](python/29_Reshape_inference.py) |
| Transpose | テンソルの次元を入れ替え | [export](python/30_Transpose_export.py) / [inference](python/30_Transpose_inference.py) |
| Flatten | テンソルを2次元に平坦化 | [export](python/31_Flatten_export.py) / [inference](python/31_Flatten_inference.py) |
| Squeeze | サイズ1の次元を削除 | [export](python/32_Squeeze_export.py) / [inference](python/32_Squeeze_inference.py) |
| Unsqueeze | サイズ1の次元を追加 | [export](python/33_Unsqueeze_export.py) / [inference](python/33_Unsqueeze_inference.py) |
| Resize | テンソルをリサイズ（拡大・縮小） | [export](python/34_Resize_export.py) / [inference](python/34_Resize_inference.py) |
| Concat | 複数のテンソルを連結 | [export](python/35_Concat_export.py) / [inference](python/35_Concat_inference.py) |
| Split | テンソルを分割 | [export](python/36_Split_export.py) / [inference](python/36_Split_inference.py) |
| Slice | テンソルの一部を切り出し | [export](python/37_Slice_export.py) / [inference](python/37_Slice_inference.py) |
| Gather | インデックスで要素を収集 | [export](python/38_Gather_export.py) / [inference](python/38_Gather_inference.py) |
| ScatterND | インデックス位置に値を散布 | [export](python/39_ScatterND_export.py) / [inference](python/39_ScatterND_inference.py) |

## 5. 線形代数 (Linear Algebra)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| MatMul | 行列乗算 | [export](python/40_MatMul_export.py) / [inference](python/40_MatMul_inference.py) |
| Gemm | 一般行列乗算 (alpha*A*B + beta*C) | [export](python/41_Gemm_export.py) / [inference](python/41_Gemm_inference.py) |

## 6. 比較演算 (Comparison Operations)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Equal | 要素ごとの等価比較 (A == B) | [export](python/42_Equal_export.py) / [inference](python/42_Equal_inference.py) |
| Greater | 要素ごとの大なり比較 (A > B) | [export](python/43_Greater_export.py) / [inference](python/43_Greater_inference.py) |
| GreaterOrEqual | 要素ごとの以上比較 (A >= B) | [export](python/44_GreaterOrEqual_export.py) / [inference](python/44_GreaterOrEqual_inference.py) |
| Less | 要素ごとの小なり比較 (A < B) | [export](python/45_Less_export.py) / [inference](python/45_Less_inference.py) |
| LessOrEqual | 要素ごとの以下比較 (A <= B) | [export](python/46_LessOrEqual_export.py) / [inference](python/46_LessOrEqual_inference.py) |

## 7. 集約・統計演算 (Reduction Operations)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| ReduceSum | 指定軸に沿った合計 | [export](python/47_ReduceSum_export.py) / [inference](python/47_ReduceSum_inference.py) |
| ReduceMean | 指定軸に沿った平均 | [export](python/48_ReduceMean_export.py) / [inference](python/48_ReduceMean_inference.py) |
| ReduceMax | 指定軸に沿った最大値 | [export](python/49_ReduceMax_export.py) / [inference](python/49_ReduceMax_inference.py) |
| ReduceMin | 指定軸に沿った最小値 | [export](python/50_ReduceMin_export.py) / [inference](python/50_ReduceMin_inference.py) |
| ReduceProd | 指定軸に沿った積 | [export](python/51_ReduceProd_export.py) / [inference](python/51_ReduceProd_inference.py) |
| ReduceL2 | L2ノルム (√Σx²) | [export](python/52_ReduceL2_export.py) / [inference](python/52_ReduceL2_inference.py) |
| ReduceL1 | L1ノルム (Σ\|x\|) | [export](python/53_ReduceL1_export.py) / [inference](python/53_ReduceL1_inference.py) |
| ReduceSumSquare | 二乗和 (Σx²) | [export](python/54_ReduceSumSquare_export.py) / [inference](python/54_ReduceSumSquare_inference.py) |
| ReduceLogSumExp | log(Σe^x)（数値安定版） | [export](python/55_ReduceLogSumExp_export.py) / [inference](python/55_ReduceLogSumExp_inference.py) |
| ReduceLogSum | log(Σx) | [export](python/56_ReduceLogSum_export.py) / [inference](python/56_ReduceLogSum_inference.py) |

## 8. ユーティリティ (Utility Operations)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| Pad | テンソルにパディングを追加 | [export](python/57_Pad_export.py) / [inference](python/57_Pad_inference.py) |

## 9. 画像処理 (Image Processing)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| SpaceToDepth | 空間次元をチャネル次元に変換 | [export](python/58_SpaceToDepth_export.py) / [inference](python/58_SpaceToDepth_inference.py) |
| DepthToSpace | チャネル次元を空間次元に変換 | [export](python/59_DepthToSpace_export.py) / [inference](python/59_DepthToSpace_inference.py) |

## 10. 制御フロー (Control Flow)

| オペレータ | 説明 | Pythonコード |
|-----------|------|-------------|
| ReverseSequence | シーケンスを指定長まで反転 | [export](python/60_ReverseSequence_export.py) / [inference](python/60_ReverseSequence_inference.py) |
