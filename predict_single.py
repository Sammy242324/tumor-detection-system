import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("body_tumor_model.h5")

class_labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

def predict_image(img_path):
 
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    print(f" Image: {img_path}")
    print(f"   ➤ Predicted: {class_labels[predicted_class]} (Confidence: {confidence * 100:.2f}%)\n")

if __name__ == "__main__":
    
    folder_path = "dataset/Testing/"

    
    supported_ext = (".jpg", ".jpeg", ".png")

  
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_ext):
            img_path = os.path.join(folder_path, filename)
            predict_image(img_path)
