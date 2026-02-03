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
<img width="278" height="676" alt="image" src="https://github.com/user-attachments/assets/af9ff605-78db-4f91-b921-2b0da71854d1" />
<img width="244" height="667" alt="image" src="https://github.com/user-attachments/assets/6b8f56f1-907b-45ba-95cd-655d0922a324" />


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

<img width="853" height="261" alt="image" src="https://github.com/user-attachments/assets/9a5f402b-8d37-491f-891c-849760db9d83" />

