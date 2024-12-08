from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from typing import Generator
import time
import axengine as axe

app = FastAPI()

def load_model(model_path):
    session = axe.InferenceSession(model_path)

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("\nModel Information:")
    print("Inputs:")
    for input in inputs:
        print(f"- Name: {input.name}")
        print(f"- Shape: {input.shape}")
#        print(f"- Type: {input.type}")
    print("\nOutputs:")
    for output in outputs:
        print(f"- Name: {output.name}")
        print(f"- Shape: {output.shape}\n\n")
#        print(f"- Type: {output.type}")

    return session

def get_top_k_predictions(output, k=5):
#    print("\nOutput tensor information:")
#    print(f"Shape: {output[0].shape}")
#    print(f"dtype: {output[0].dtype}")
    print(f"Value range: [{output[0].min():.3f}, {output[0].max():.3f}]")

    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores


def resize_frame(frame: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_image(img, target_size=(256, 256), crop_size=(224, 224)):
    if img is None:
        raise ValueError(f"Failed to load image")
    
    # Convert BGR to RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    height, width = img.shape[:2]

    # Determine the shorter side and calculate the center crop
    if width < height:
        crop_area = width
    else:
        crop_area = height

    crop_x = (width - crop_area) // 2
    crop_y = (height - crop_area) // 2

    # Crop the center square
    img = img[crop_y:crop_y + crop_area, crop_x:crop_x + crop_area]

    # Resize the image to target_size
    img = cv2.resize(img, target_size)

    # Crop the center to crop_size
    crop_x = (target_size[0] - crop_size[0]) // 2
    crop_y = (target_size[1] - crop_size[1]) // 2
    img = img[crop_y:crop_y + crop_size[1], crop_x:crop_x + crop_size[0]]

    # Convert to numpy array (it's already a numpy array with OpenCV)
    #img_array = img.astype("uint8")
    
    # Flip the color channels back (equivalent to [..., ::-1])
    #img_array = img_array[..., ::-1]
    img_array = img[..., ::-1]
#    print("Final tensor shape:", img_array.shape)

    # Debug: Print preprocessed tensor information
#    print("\nPreprocessed tensor information:")
#    print(f"Shape: {img_array.shape}")
#    print(f"dtype: {img_array.dtype}")
#    print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    return img_array


def get_video_stream() -> Generator[bytes, None, None]:
    print(f"please access video stream")

    session = load_model("/opt/data/npu/models/mobilenetv2.axmodel")
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = resize_frame(frame)

            target_size=(256, 256)
            crop_size=(224, 224)
            input_tensor = preprocess_image(frame, target_size, crop_size)

        # Measure inference time
            start_time = time.time()
            output = session.run(None, {input_name: input_tensor})
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
            print(f"Inference Time: {inference_time:.2f} ms")
        
        # Get top k predictions
            k=5
            top_k_indices, top_k_scores = get_top_k_predictions(output, k)
        
        # Print results for this iteration
            print(f"Top {k} Predictions:")
            for j in range(k):
                print(f"Class Index: {top_k_indices[j]}, Score: {top_k_scores[j]}")

            time.sleep(0.005)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(
        get_video_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)

