# Body Tumor Detection (Demo)

Minimal Keras baseline for binary classification (tumor vs normal) and a Streamlit web form to upload an image and visualize Grad‑CAM heatmaps.

## Project Structure
```
body-tumor-detection/
  app.py
  train.py
  infer.py
  utils.py
  requirements.txt
  data/
    train/{tumor,normal}
    val/{tumor,normal}
  models/
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training
Put images into `data/train/tumor`, `data/train/normal`, and validation into `data/val/...`
```bash
python train.py --epochs 10 --img-size 224 --batch-size 32
# Optional fine-tuning
python train.py --epochs 10 --finetune --ft-epochs 5
```

## Run the Streamlit App
```bash
streamlit run app.py
```

## CLI Inference
```bash
python infer.py --image path/to/image.jpg
```

## Notes
- Works on generic RGB images (PNG/JPG). Convert DICOM to PNG/JPG first.
- Grad-CAM last conv layer name for MobileNetV2: `Conv_1`.
- This is **not** a medical device; educational use only.
- 
<img width="849" height="738" alt="image" src="https://github.com/user-attachments/assets/22af484a-d794-4bb1-a7b9-0449e3e13187" />
