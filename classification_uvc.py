# Copyright (c) 2024 nnn112358

import axengine as axe
import numpy as np
import cv2
import time 

def load_model(model_path):
    session = axe.InferenceSession(model_path)

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("\nModel Information:")
    print("Inputs:")
    for input in inputs:
        print(f"- Name: {input.name}")
        print(f"- Shape: {input.shape}")
    print("\nOutputs:")
    for output in outputs:
        print(f"- Name: {output.name}")
        print(f"- Shape: {output.shape}")

    return session

def preprocess_frame(frame, target_size=(256, 256), crop_size=(224, 224)):
    # Frame is already in BGR format from OpenCV
    
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    # Convert to numpy array
    img_array = img.astype("uint8")
    
    # Flip the color channels back
    img_array = img_array[..., ::-1]

    return img_array

def get_top_k_predictions(output, k=5):
    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores

def main(model_path, target_size, crop_size, k):
    # Load the model
    session = load_model(model_path)
    input_name = session.get_inputs()[0].name

    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    if not cap.isOpened():
        raise ValueError("Failed to open camera")

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display the original frame
            cv2.imshow('Original', frame)

            # Preprocess the frame
            input_tensor = preprocess_frame(frame, target_size, crop_size)

            # Measure inference time
            start_time = time.time()
            output = session.run(None, {input_name: input_tensor})
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get top k predictions
            top_k_indices, top_k_scores = get_top_k_predictions(output, k)

            # Create a copy of the frame for displaying results
            result_frame = frame.copy()
            
            # Draw results on the frame
            y_position = 30
            cv2.putText(result_frame, f"Inference Time: {inference_time:.2f} ms", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for i in range(k):
                y_position += 30
                text = f"Class {top_k_indices[i]}: {top_k_scores[i]:.4f}"
                cv2.putText(result_frame, text, (10, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the result frame
            cv2.imshow('Results', result_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "/opt/data/npu/models/mobilenetv2.axmodel"
    TARGET_SIZE = (224, 224)
    CROP_SIZE = (224, 224)
    K = 5  # Top K predictions
    main(MODEL_PATH, TARGET_SIZE, CROP_SIZE, K)