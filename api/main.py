from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer
import io, json
import pathlib

app = FastAPI()

SAVED_MODEL_DIR = pathlib.Path("/mnt/d/04_Food101-EfficientNetV2S-model/models/export/food101_savedmodel")
CLASS_NAMES_DIR = pathlib.Path("/mnt/d/04_Food101-EfficientNetV2S-model/Phase-C_Evaluation-Analysis/class_names.json")

model = tf.keras.Sequential([TFSMLayer(SAVED_MODEL_DIR.as_posix(), call_endpoint="serving_default")])

with open(CLASS_NAMES_DIR) as f:
    class_names = json.load(f)

IMG_SIZE = (384, 384)

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype("float32")
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img[None, ...]

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = preprocess(image)

    outputs = model(x)
    probs = outputs["output_layer"].numpy()[0]

    top5_idx = probs.argsort()[-5:][::-1]

    return {
        "top1": {
            "label": class_names[top5_idx[0]],
            "confidence": float(probs[top5_idx[0]])
        },
        "top5": [
            {
                "label": class_names[i],
                "confidence": float(probs[i])
            } for i in top5_idx
        ]
    }