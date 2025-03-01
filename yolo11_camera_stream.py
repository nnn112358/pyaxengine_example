from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import axengine as axe
import time
from typing import Generator, List
from dataclasses import dataclass

app = FastAPI()

@dataclass
class Detection:
    bbox: List[float]
    label: int
    prob: float

# 定数の設定
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45

#M5Stackのllm-yolo_1.4-m5stack1_arm64.deb内のaxmodelを使う場合
INPUT_SIZE = (320, 320)
#AXERAのサンプルファイルを使用する場合
#INPUT_SIZE = (640, 640)


REG_MAX = 16
MODEL_PATH = '/opt/m5stack/data/yolo11n/yolo11n.axmodel'

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def initialize_model(model_path: str) -> axe.InferenceSession:
    """モデルを初期化する"""
    session = axe.InferenceSession(model_path)
    return session

def process_frame(frame: np.ndarray, target_size=(320, 320)) -> np.ndarray:
    """フレームの前処理を行う"""

    #M5Stackのllm-yolo_1.4-m5stack1_arm64.deb内のaxmodelを使う場合
    # BGRからRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #AXERAのサンプルファイルのaxmodelを使用する場合,COLOR_BGR2RGBは不要。
    
    # リサイズ
    resized = cv2.resize(rgb_frame, target_size)
    # バッチ次元を追加
    return np.expand_dims(resized, axis=0).astype(np.uint8)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """数値的に安定なソフトマックスの実装"""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def decode_predictions(feat: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """Distribution Focal Lossの出力をデコード"""
    prob = softmax(feat, axis=-1)
    return np.sum(prob * np.arange(reg_max), axis=-1)

def process_detections(outputs: List[np.ndarray], original_shape: tuple, 
                      input_size: tuple) -> List[Detection]:
    """検出結果の後処理を行う"""
    detections = []
    num_classes = 80
    bbox_channels = 4 * REG_MAX

    heads = [
        {'output': outputs[0], 'stride': 8},
        {'output': outputs[1], 'stride': 16},
        {'output': outputs[2], 'stride': 32}
    ]

    for head in heads:
        output = head['output']
        stride = head['stride']
        
        batch_size, grid_h, grid_w, channels = output.shape
        bbox_part = output[:, :, :, :bbox_channels]
        class_part = output[:, :, :, bbox_channels:]

        bbox_part = bbox_part.reshape(grid_h * grid_w, 4, REG_MAX)
        class_part = class_part.reshape(batch_size, grid_h * grid_w, num_classes)

        for i in range(grid_h * grid_w):
            h = i // grid_w
            w = i % grid_w
            
            class_scores = class_part[0, i, :]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            box_prob = 1 / (1 + np.exp(-class_score))

            if box_prob < CONFIDENCE_THRESHOLD:
                continue

            bbox = bbox_part[i, :, :]
            dfl_scores = [decode_predictions(bbox[j, :]) for j in range(4)]
            
            pb_cx = (w + 0.5) * stride
            pb_cy = (h + 0.5) * stride

            x0 = pb_cx - dfl_scores[0] * stride
            y0 = pb_cy - dfl_scores[1] * stride
            x1 = pb_cx + dfl_scores[2] * stride
            y1 = pb_cy + dfl_scores[3] * stride

            # 座標のスケーリング
            scale_x = original_shape[1] / input_size[0]
            scale_y = original_shape[0] / input_size[1]
            x0 = np.clip(x0 * scale_x, 0, original_shape[1])
            y0 = np.clip(y0 * scale_y, 0, original_shape[0])
            x1 = np.clip(x1 * scale_x, 0, original_shape[1])
            y1 = np.clip(y1 * scale_y, 0, original_shape[0])

            width = x1 - x0
            height = y1 - y0

            detections.append(Detection(
                bbox=[float(x0), float(y0), float(width), float(height)],
                label=int(class_id),
                prob=float(box_prob)
            ))

    return detections

def apply_nms(detections: List[Detection]) -> List[Detection]:
    """非最大値抑制を適用"""
    if not detections:
        return []

    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.prob for d in detections])
    class_ids = np.array([d.label for d in detections])

    final_detections = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 0] + cls_boxes[:, 2]
        y2 = cls_boxes[:, 1] + cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = cls_scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= NMS_THRESHOLD)[0]
            order = order[inds + 1]
            
        for idx in keep:
            final_detections.append(Detection(
                bbox=cls_boxes[idx].tolist(),
                label=int(cls),
                prob=float(cls_scores[idx])
            ))

    return final_detections

def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """検出結果を画像上に描画"""
    for det in detections:
        x, y, w, h = map(int, det.bbox)
        label = f"{COCO_CLASSES[det.label]}: {det.prob:.2f}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def get_video_stream() -> Generator[bytes, None, None]:
    """ビデオストリームを生成"""
    session = initialize_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            original_shape = frame.shape[:2]
            input_tensor = process_frame(frame, INPUT_SIZE)
            
            # 推論実行
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # 検出実行
            detections = process_detections(outputs, original_shape, INPUT_SIZE)
            detections = apply_nms(detections)
            
            # 結果の描画
            result_frame = draw_detections(frame, detections)
            
            # JPEG形式でエンコード
            _, buffer = cv2.imencode('.jpg', result_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.01)  # フレームレート調整
            
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
