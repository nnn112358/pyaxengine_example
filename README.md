## abst
M5StackのModule-LLMで、
pyaxengineのpython＋opencvを動かすサンプルです。

## Install

Module-LLMにルートでログインし、pyaxengineをインストールします。
```
root@m5stack-LLM # wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc1/axengine-0.0.1-py3-none-any.whl
root@m5stack-LLM # pip install axengine-0.0.1-py3-none-any.whl
```
Module-LLMにルートでログインし、 fastapi opencv-python uvicornをインストールします。
```
root@m5stack-LLM # pip install fastapi opencv-python uvicorn
```
## Example

```
root@m5stack-LLM:# python3 classification_opencv.py
→画像ファイルから画像を取得し、mobilenet v2 のクラス分類をかけるサンプルです。

root@m5stack-LLM:# python3 classification_camera_stream.py
→USBカメラから画像を取得し、mobilenet v2 のクラス分類をかけた後、httpにストリーミングを行うサンプルです。
 Pythonが起動した後、 ブラウザ で　http://{Module-LLMのアドレス}:7777/video　にアクセスします。

root@m5stack-LLM:# python3 depth_anything_camera_stream.py
→USBカメラから画像を取得し、depth_anythingをかけた後、httpにストリーミングを行うサンプルです。
 Pythonが起動した後、 ブラウザ で　http://{Module-LLMのアドレス}:7777/video　にアクセスします。
```





## 参考

Module-LLMにUSBカメラを接続する <br>
https://qiita.com/nnn112358/items/b284763de0d333dfdb73 <br>
 <br>
pyaxengine <br>
https://github.com/AXERA-TECH/pyaxengine <br>
