import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Load model (use your trained model path here)
model = tf.keras.models.load_model('body_tumor_model_mobilenetv2.h5')

# Preprocess image function
def preprocess_image(img_path, img_size=224):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # MobileNetV2 scaling [-1,1]
    return x, img

# Grad-CAM heatmap function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index.numpy()

# Overlay heatmap on original image
def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (orig_img.width, orig_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    orig_img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_img_cv, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return Image.fromarray(overlay_rgb)

# Example usage
img_path = "your_mri_image.jpg"  # replace with your image
img_array, orig_img = preprocess_image(img_path)

# The name of the last conv layer in MobileNetV2 is 'Conv_1_relu' or similar - verify with model.summary()
last_conv_layer_name = "Conv_1_relu"

heatmap, pred_class = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

overlay_img = overlay_heatmap_on_image(orig_img, heatmap)
overlay_img.show()

# Print prediction
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # update with your classes
print(f"Predicted class: {class_names[pred_class]}")
