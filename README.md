## abst
pyaxengineのpythonテストファイルです。
opencv対応しました。

## 注記
post処理がC++版と異なる。

## Install

```
wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc1/axengine-0.0.1-py3-none-any.whl
pip install axengine-0.0.1-py3-none-any.whl
```

## mobilenetv2 class

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
```

### mobilenetv2.axmodel
![image](https://github.com/user-attachments/assets/e7528c19-3f1e-4c49-868d-89003c747664)



## 参考
pyaxengine
https://github.com/AXERA-TECH/pyaxengine
