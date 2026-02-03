import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = "sign_language_model.h5"  # Change this to your trained model file
TEST_DIR = os.path.join("data", "test")  # Updated to use "data/test"

# ========================
# CHECK IF TEST FOLDER EXISTS
# ========================
if not os.path.exists(TEST_DIR):
    print(f"Error: Test folder not found at {TEST_DIR}")
    print("Please create the folder and add test images in subfolders: healthy, normal, tumor.")
    exit()

# ========================
# LOAD TRAINED MODEL
# ========================
print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ========================
# IMAGE DATA GENERATOR FOR TEST DATA
# ========================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),  # Match this with your model input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ========================
# EVALUATE MODEL
# ========================
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}\n")

# ========================
# PREDICTIONS
# ========================
print("Generating predictions...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Map indices to class names
class_indices = {v: k for k, v in test_generator.class_indices.items()}

print("\nSample Predictions:")
for i in range(len(predicted_classes)):
    file_name = test_generator.filenames[i]
    predicted_label = class_indices[predicted_classes[i]]
    print(f"Image: {file_name} --> Predicted: {predicted_label}")

print("\nTesting complete!")
