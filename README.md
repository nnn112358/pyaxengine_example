## abst
pyaxengineのpythonテストファイルです。


## Install

```
wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc1/axengine-0.0.1-py3-none-any.whl
pip install axengine-0.0.1-py3-none-any.whl
```

## mobilenetv2 class
![image](https://github.com/user-attachments/assets/e7528c19-3f1e-4c49-868d-89003c747664)

```
root@m5stack-LLM:# python3 ./classification_opencv.py
[INFO] Chip type: ChipType.MC20E
[INFO] Engine version: 2.6.3sp
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Model type: 0 (half core)
[INFO] Compiler version: 1.8-beta1 6a7e59de

Model Information:
Inputs:
- Name: input
- Shape: [1, 224, 224, 3]

Outputs:
- Name: output
- Shape: [1, 1000]
224
224
Final tensor shape: (224, 224, 3)

Preprocessed tensor information:
Shape: (224, 224, 3)
dtype: uint8
Value range: [0.000, 255.000]
Input name: input
Input tensor shape: (224, 224, 3)
Expected input shape: [1, 224, 224, 3]
Expected output shape: [1, 1000]

Iteration 1/10
Inference Time: 27.43 ms

Output tensor information:
Shape: (1, 1000)
dtype: float32
Value range: [-6.340, 11.226]
Top 5 Predictions:
Class Index: 287, Score: 11.226346969604492
Class Index: 281, Score: 11.226346969604492
Class Index: 282, Score: 11.09427261352539
Class Index: 277, Score: 10.830123901367188
Class Index: 285, Score: 10.830123901367188

Iteration 2/10
Inference Time: 2.00 ms

Output tensor information:
Shape: (1, 1000)
dtype: float32
Value range: [-6.340, 11.226]
Top 5 Predictions:
Class Index: 287, Score: 11.226346969604492
Class Index: 281, Score: 11.226346969604492
Class Index: 282, Score: 11.09427261352539
Class Index: 277, Score: 10.830123901367188
Class Index: 285, Score: 10.830123901367188

Iteration 3/10
Inference Time: 1.94 ms

Output tensor information:
Shape: (1, 1000)
dtype: float32
Value range: [-6.340, 11.226]
Top 5 Predictions:
Class Index: 287, Score: 11.226346969604492
Class Index: 281, Score: 11.226346969604492
Class Index: 282, Score: 11.09427261352539
Class Index: 277, Score: 10.830123901367188
Class Index: 285, Score: 10.830123901367188

```


## 参考
pyaxengine
https://github.com/AXERA-TECH/pyaxengine
