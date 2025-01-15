from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from typing import Generator
import time
import axengine as axe
from typing import List

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
    print("\nOutput tensor information:")
    print(f"Shape: {output[0].shape}")
    print(f"dtype: {output[0].dtype}")
#    print(f"Value range: [{output[0].min():.3f}, {output[0].max():.3f}]")

    # Get top k predictions
#    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
#    top_k_scores = output[0].flatten()[top_k_indices]
    return 0

def resize_frame(frame: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_image(img, target_size=(384, 256)):
    if img is None:
        raise ValueError(f"Failed to load image")
    
    # Convert BGR to RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    height, width = img.shape[:2]

    # Resize the image to target_size
    img = cv2.resize(img, target_size)

    img_array = img[..., ::-1]
#    print("Final tensor shape:", img_array.shape)

    # Debug: Print preprocessed tensor information
#    print("\nPreprocessed tensor information:")
#    print(f"Shape: {img_array.shape}")
#    print(f"dtype: {img_array.dtype}")
#    print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def post_process(output_data: np.ndarray,original_image: np.ndarray):
    """
    Post-process depth estimation output and save visualization results.
    """

    output_data2 = np.array(output_data)

    print(f"Shape: {output_data2.shape}")


    # Reshape feature map if needed
    feature = output_data2.reshape(output_data2.shape[-2:])
    
    # Normalize feature map
    min_val = np.min(feature)
    max_val = np.max(feature)
    feature = (feature - min_val) / (max_val - min_val)
    
    # Scale to 8-bit
    feature = (feature * 255).astype(np.uint8)
    
    # Apply colormap
    colored_depth = cv2.applyColorMap(feature, cv2.COLORMAP_MAGMA)
    
    # Resize to original image dimensions
    colored_depth = cv2.resize(colored_depth, (original_image.shape[1], original_image.shape[0]))
    combined_image = np.concatenate([original_image, colored_depth], axis=1)
    return combined_image


def get_video_stream() -> Generator[bytes, None, None]:
    print(f"please access video stream")

    session = load_model("./depth_anything_u8.axmodel")
    target_size=(384, 256)

    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = resize_frame(frame)

            input_tensor = preprocess_image(frame, target_size)

        # Measure inference time
            start_time = time.time()
            output = session.run(None, {input_name: input_tensor})
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
            print(f"Inference Time: {inference_time:.2f} ms")
        
            out_img=post_process(output,frame)
 
            time.sleep(0.005)
            _, buffer = cv2.imencode('.jpg', out_img)
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

