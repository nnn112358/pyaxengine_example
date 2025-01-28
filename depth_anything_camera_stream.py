from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import axengine as axe
import time

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/depth_anything_u8.axmodel'

def initialize_model(model_path: str):
    """モデルを初期化し、推論セッションを返す"""
    return axe.InferenceSession(model_path)

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    """入力フレームの前処理を行う"""
    if frame is None:
        raise ValueError("フレームの読み込みに失敗しました")
    
    resized_frame = cv2.resize(frame, target_size)
    return np.expand_dims(resized_frame[..., ::-1], axis=0)

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    """深度マップの可視化を行う"""
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    
    # 正規化と色付け
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min())
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # 元の画像サイズにリサイズ
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    
    # 元画像と深度マップを横に並べる
    return np.concatenate([original_frame, depth_resized], axis=1)

def get_video_stream():
    """ビデオストリームを生成する"""
    session = initialize_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # フレームをリサイズして処理
            frame = cv2.resize(frame, (320, 240))
            input_tensor = process_frame(frame)
            
            # 深度推定を実行
            output = session.run(None, {input_name: input_tensor})
            
            # 結果を可視化
            visualization = create_depth_visualization(output[0], frame)
            
            # JPEGエンコードしてストリーミング
            _, buffer = cv2.imencode('.jpg', visualization)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.005)
    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    """ビデオストリーミングのエンドポイント"""
    return StreamingResponse(
        get_video_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)