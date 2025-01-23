import cv2
import numpy as np
#import onnxruntime as ort
import axengine as axe
import threading
from queue import Queue
import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import platform

class InferenceEngine:
    def __init__(self, model_path, label_path=None):
        self.input_data = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
        self.session = axe.InferenceSession(model_path)
#        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.labels = None
        if label_path:
            with open(label_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        print("\nModel Information:")
        print("Inputs:")
        for input in inputs:
            print(f"- Name: {input.name}")
            print(f"- Shape: {input.shape}")
            print(f"- Type: {input.dtype}")
        print("\nOutputs:")
        for output in outputs:
            print(f"- Name: {output.name}")
            print(f"- Shape: {output.shape}\n\n")
            print(f"- Type: {output.dtype}")


    def preprocess_frame(self, frame, target_size=(224, 224)):
        start_time = time.time()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        #img = (img / 255.0).astype(np.float32)
        #img = ((img - self.mean) / self.std).astype(np.float32)
        #img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        #print(f"Shape: {img.shape}")
        #print(f"dtype: {img.dtype}")

        self.preprocess_time = (time.time() - start_time) * 1000  # ms単位
        return img

    def run_inference(self, input_tensor):
        start_time = time.time()
        output = self.session.run(None, {self.input_name: input_tensor})
        self.inference_time = (time.time() - start_time) * 1000
        return output

    def postprocess_output(self, output, k=5):
        start_time = time.time()
        top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
        top_k_scores = output[0].flatten()[top_k_indices]
        result = list(zip([self.labels[i] for i in top_k_indices], top_k_scores)
                      ) if self.labels else list(zip(top_k_indices, top_k_scores))
        self.postprocess_time = (time.time() - start_time) * 1000
        return result

    def main_process(self, input_tensor, k=5):
        output = self.run_inference(input_tensor)
        return self.postprocess_output(output, k)


class CameraThread(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.camera_read_time = 0  # Add timing attribute
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            self.camera_read_time = (time.time() - start_time) * 1000  # Convert to ms

            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put((frame, self.camera_read_time))
            time.sleep(0.05)

        cap.release()

    def stop(self):
        self.running = False


class InferenceThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, engine):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.engine = engine
        self.running = True

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame, camera_read_time = self.frame_queue.get()  # タプルから2つの値を取得
                input_tensor = self.engine.preprocess_frame(frame)
                predictions = self.engine.main_process(input_tensor)
                if self.result_queue.qsize() < 2:
                    self.result_queue.put((frame, predictions, camera_read_time))  # camera_read_timeを追加

            time.sleep(0.01)

    def stop(self):
        self.running = False


class DisplayThread(threading.Thread):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True
        self.latest_frame = None
        self.latest_predictions = None
        self.engine = None  # Add engine attribute

    def run(self):
        while self.running:
            if not self.result_queue.empty():
                frame, predictions, camera_read_time = self.result_queue.get()  # 3つの値を取得
                self.latest_frame = frame.copy()
                self.latest_predictions = predictions
                self.latest_camera_read_time = camera_read_time  # 保存

                # Process frame
                frame = self.process_frame(frame.copy(), predictions)

                if platform.machine() == 'x86_64':
                    cv2.imshow("Camera Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
            time.sleep(0.03)
        cv2.destroyAllWindows()

    def process_frame(self, frame, predictions):
        # Draw predictions
        for i, (label, score) in enumerate(predictions):
            text = f"{label}: {score:.3f}"
            cv2.putText(frame, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add timing information
        timings = [
            f"Camera read: {self.latest_camera_read_time:.1f}ms",
            f"Preprocess: {self.engine.preprocess_time:.1f}ms",
            f"Inference: {self.engine.inference_time:.1f}ms",
            f"Postprocess: {self.engine.postprocess_time:.1f}ms"
        ]
        for i, timing in enumerate(timings):
            y_pos = frame.shape[0] - 30 * (len(timings) - i)
            cv2.putText(frame, timing, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame

    def get_latest_frame(self):
        if self.latest_frame is not None:
            return self.process_frame(self.latest_frame.copy(), self.latest_predictions)
        return None

    def stop(self):
        self.running = False


display_thread = None  # グローバル変数として保持


def generate_frames():
    while True:
        if display_thread and display_thread.latest_frame is not None:
            frame = display_thread.get_latest_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

app = FastAPI()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Camera Stream</title>
    </head>
    <body>
        <h1>Camera Stream</h1>
        <img src="/video_feed">
    </body>
    </html>
    """

def main():
    global display_thread
    MODEL_PATH = "model/mobilenetv2.axmodel"
    LABEL_PATH = "model/imagenet_labels.txt"

    engine = InferenceEngine(MODEL_PATH, LABEL_PATH)
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    camera_thread = CameraThread(frame_queue)
    inference_thread = InferenceThread(frame_queue, result_queue, engine)
    display_thread = DisplayThread(result_queue)

    display_thread.engine = engine

    camera_thread.start()
    inference_thread.start()
    display_thread.start()

    def shutdown_handler():
        print("\nShutting down gracefully...")
        camera_thread.stop()
        inference_thread.stop()
        display_thread.stop()

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                pass

        camera_thread.join(timeout=1)
        inference_thread.join(timeout=1)
        display_thread.join(timeout=1)

    import uvicorn
    import atexit
    atexit.register(shutdown_handler)

    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        shutdown_handler()

if __name__ == "__main__":
    main()