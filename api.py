# Example FastAPI deployment route for inference
from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from model import get_2p5d_model
from PIL import Image
import io

app = FastAPI()

# Load model (adjust path as needed)
model = get_2p5d_model(256, 256, n_ch=3, deep_supervision=True, pretrained=False)
model.load_weights("final_model.h5")

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dim
    preds = model.predict(arr)[0]
    mask = (preds > 0.5).astype(np.uint8)[0, ..., 0]
    return {"mask": mask.tolist()}
