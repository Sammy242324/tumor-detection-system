import os
import uuid
import datetime
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
REPORT_FOLDER = 'reports'
HISTORY_FILE = 'history.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

MODEL_PATH = "body_tumor_model.h5"
model = load_model(MODEL_PATH)

def process_image(file_path, output_path):
    input_shape = model.input_shape
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    color_mode = 'grayscale' if channels == 1 else 'rgb'

    img = image.load_img(file_path, target_size=(height, width), color_mode=color_mode)
    img_array = image.img_to_array(img) / 255.0
    if channels == 1 and img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print("Raw prediction output:", prediction)  

    output_neurons = prediction.shape[1]
    if output_neurons == 1:
      
        confidence = float(prediction[0][0])
        label = "Tumor Detected" if confidence >= 0.5 else "No Tumor"
    elif output_neurons == 2:
        
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        label = "No Tumor" if predicted_class == 0 else "Tumor Detected"
    else:
     
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        label = f"Class {predicted_class} Detected"

    original_img = cv2.imread(file_path)
    color = (0, 255, 0) if label == "No Tumor" else (0, 0, 255)
    cv2.putText(original_img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(original_img, (20, 20),
                  (original_img.shape[1]-20, original_img.shape[0]-20),
                  color, 3)
    cv2.imwrite(output_path, original_img)

    return label, round(confidence, 4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Error: No file part in request", 400
    file = request.files['file']
    if file.filename == '':
        return "Error: No file selected", 400
    patient_name = request.form.get('name')
    patient_age = request.form.get('age')
    patient_gender = request.form.get('gender')
    if not all([patient_name, patient_age, patient_gender]):
        return "Error: Missing patient details", 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    label, confidence = process_image(file_path, result_path)

    patient_data = {
        "name": patient_name,
        "age": patient_age,
        "gender": patient_gender,
        "result_img": result_filename,
        "prediction": label,
        "confidence": confidence,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    history.append(patient_data)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

    return render_template('result.html', patient=patient_data)

@app.route('/generate_report', methods=['POST'])
def generate_report():
 
    patient_name = request.form.get('name')
    patient_age = request.form.get('age')
    patient_gender = request.form.get('gender')
    prediction = request.form.get('prediction')
    confidence = float(request.form.get('confidence'))
    result_img = request.form.get('result_img')
    date = request.form.get('date')


    pdf_filename = f"report_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor("#007bff"))
    c.drawCentredString(width / 2, height - 50, "Smart Health Diagnostic Center")
    c.setFont("Helvetica-Oblique", 12)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, height - 70, "Automated Tumor Detection Report")

    
    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Information:")
    c.setFont("Helvetica", 11)
    y -= 20
    c.drawString(70, y, f"Name: {patient_name}")
    y -= 15
    c.drawString(70, y, f"Age: {patient_age}")
    y -= 15
    c.drawString(70, y, f"Gender: {patient_gender}")
    y -= 15
    c.drawString(70, y, f"Date: {date}")

    # Diagnosis Result
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Diagnosis Result:")
    c.setFont("Helvetica", 11)
    y -= 20
    bar_color = colors.green if prediction == "No Tumor" else colors.red
    c.setFillColor(bar_color)
    c.drawString(70, y, f"Prediction: {prediction}")
    c.setFillColor(colors.black)
    y -= 15
    c.drawString(70, y, f"Confidence: {confidence*100:.2f}%")
    y -= 20
    bar_x, bar_y = 70, y
    bar_width, bar_height = 400, 15
    c.setStrokeColor(colors.black)
    c.rect(bar_x, bar_y, bar_width, bar_height, stroke=1, fill=0)
    c.setFillColor(bar_color)
    c.rect(bar_x, bar_y, bar_width * confidence, bar_height, stroke=0, fill=1)

    y -= 180
    img_path = os.path.join(RESULT_FOLDER, result_img)
    if os.path.exists(img_path):
        c.drawImage(img_path, 120, y, width=300, height=200, preserveAspectRatio=True)

    y -= 60
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(50, y, "Explanation & Next Steps:")
    c.setFont("Helvetica", 10)
    y -= 20
    if prediction == "No Tumor":
        c.drawString(70, y, "- No tumor was detected in the scan.")
        y -= 15
        c.drawString(70, y, "- However, regular check-ups are recommended.")
    else:
        c.drawString(70, y, "- A tumor-like structure was detected in the scan.")
        y -= 15
        c.drawString(70, y, "- Please consult a certified oncologist for further tests.")
        y -= 15
        c.drawString(70, y, "- Additional imaging (MRI, PET scan) may be advised.")

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.gray)
    c.drawCentredString(width / 2, 50,
        "This report is system-generated and must be verified by a certified medical professional.")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(width - 50, 100, "Doctor’s Signature: _______________")

    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route('/history')
def history():
    with open(HISTORY_FILE, 'r') as f:
        history_data = json.load(f)
    return render_template('history.html', history=history_data)

if __name__ == '__main__':
    print("\nServer is running! Open http://127.0.0.1:5000 in your browser.\n")
    app.run(debug=True)
