# infer.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "body_tumor_model.h5"
IMAGE_SIZE = (224, 224)  


model = load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_tumor(image_path):
   
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

   
    class_names = ["Brain Tumor", "Lung Tumor", "Normal"]  

    print(f"Predicted Class: {class_names[class_idx]}")
    print(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    test_image_path = "test_image.jpg" 
    if os.path.exists(test_image_path):
        predict_tumor(test_image_path)
    else:
        print("Error: Test image not found!")
