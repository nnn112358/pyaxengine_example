## abst
pyaxengineのpython＋opencvのサンプルです。

## Install

```
wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc1/axengine-0.0.1-py3-none-any.whl
pip install axengine-0.0.1-py3-none-any.whl
```

```
root@m5stack-LLM:# python3 classification_opencv.py
→画像ファイルから画像を取得し、mobilenet v2 のクラス分類をかけるサンプルです。

root@m5stack-LLM:# python3 classification_camera_stream.py
→USBカメラから画像を取得し、mobilenet v2 のクラス分類をかけた後、httpにストリーミングを行うサンプルです。

root@m5stack-LLM:# python3 depth_anything_camera_stream.py
→USBカメラから画像を取得し、depth_anythingをかけた後、httpにストリーミングを行うサンプルです。
```



### mobilenetv2.axmodel
![image](https://github.com/user-attachments/assets/e7528c19-3f1e-4c49-868d-89003c747664)

### depth_anything.axmodel
![image](https://github.com/user-attachments/assets/13aa0267-330e-45d8-87bd-f59cb4e51967)


## 参考

Module-LLMにUSBカメラを接続する
https://qiita.com/nnn112358/items/b284763de0d333dfdb73

pyaxengine
https://github.com/AXERA-TECH/pyaxengine
