# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

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
#        print(f"- Type: {input.type}")
    print("\nOutputs:")
    for output in outputs:
        print(f"- Name: {output.name}")
        print(f"- Shape: {output.shape}")
#        print(f"- Type: {output.type}")

    return session

def preprocess_image(image_path, target_size=(256, 256), crop_size=(224, 224)):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    height, width = img.shape[:2]
    
    print(width)
    print(height)

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
    img_array = img.astype("uint8")
    
    # Flip the color channels back (equivalent to [..., ::-1])
    img_array = img_array[..., ::-1]
    print("Final tensor shape:", img_array.shape)

    # Debug: Print preprocessed tensor information
    print("\nPreprocessed tensor information:")
    print(f"Shape: {img_array.shape}")
    print(f"dtype: {img_array.dtype}")
    print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    return img_array

def get_top_k_predictions(output, k=5):
    print("\nOutput tensor information:")
    print(f"Shape: {output[0].shape}")
    print(f"dtype: {output[0].dtype}")
    print(f"Value range: [{output[0].min():.3f}, {output[0].max():.3f}]")

    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores

def main(model_path, image_path, target_size, crop_size, k):
    # Load the model
    session = load_model(model_path)

    # Preprocess the image
    input_tensor = preprocess_image(image_path, target_size, crop_size)

    # Get input name and run inference
    input_name = session.get_inputs()[0].name


    print("Input name:", input_name)
    print("Input tensor shape:", input_tensor.shape)
    print("Expected input shape:", session.get_inputs()[0].shape)
    print("Expected output shape:", session.get_outputs()[0].shape)
  
    for i in range(100):
        print(f"\nIteration {i+1}/10")
        
        # Measure inference time
        start_time = time.time()
        output = session.run(None, {input_name: input_tensor})
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Inference Time: {inference_time:.2f} ms")
        
        # Get top k predictions
        top_k_indices, top_k_scores = get_top_k_predictions(output, k)
        
        # Print results for this iteration
        print(f"Top {k} Predictions:")
        for j in range(k):
            print(f"Class Index: {top_k_indices[j]}, Score: {top_k_scores[j]}")





if __name__ == "__main__":
    MODEL_PATH = "/opt/data/npu/models/mobilenetv2.axmodel"
    IMAGE_PATH = "./cat.jpg"
    TARGET_SIZE = (224, 224)  # Resize to 256x256
    CROP_SIZE = (224, 224)  # Crop to 224x224
    K = 5  # Top K predictions
    main(MODEL_PATH, IMAGE_PATH, TARGET_SIZE, CROP_SIZE, K)






