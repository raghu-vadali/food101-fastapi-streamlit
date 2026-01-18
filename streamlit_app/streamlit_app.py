import streamlit as st
import requests
from PIL import Image
from collections import defaultdict
import pandas as pd
import numpy as np
import json

st.title("Food101 Image Classification")


#------Sidebar controls (confidence + Top-K)----------------------------------
st.sidebar.header("Prediction Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

top_k = st.sidebar.selectbox(
    "Top-K predictions",
    options=[1, 3, 5],
    index=2
)


#------Live Predictions (uses FastAPI)---------------------------------------
st.header("Live Food Classification")

uploaded = st.file_uploader("Upload a food image")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image")

    res = requests.post(
        "http://localhost:8000/predict",
        files={"file": uploaded.getvalue()}
    )

    if res.status_code == 200:
        data = res.json()

        st.subheader("Predictions")

        for pred in data["top5"][:top_k]:
            if pred["confidence"] >= confidence_threshold:
                st.write(
                    f"**{pred['label']}** — {pred['confidence']:.2f}"
                )
    else:
        st.error(res.text)
        
        
#------Global Metrics (cached)-----------------------------------------------   
st.header("Model Performance")

with open("/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/metrics.json") as f:
    metrics = json.load(f)

col1, col2 = st.columns(2)
col1.metric("Top-1 Accuracy", f"{metrics['top1_accuracy']:.2%}")
col2.metric("Top-5 Accuracy", f"{metrics['top5_accuracy']:.2%}")


#------Per-Class Weaknesses/ Strengths (bar chart)--------------------------------------
st.header("Per-Class Weaknesses")

y_true = np.load("/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/y_true.npy")
y_pred = np.load("/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/y_pred.npy")
y_probs = np.load("/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/y_probs.npy")

with open("/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/class_names.json") as f:
    class_names = json.load(f)

class_correct = defaultdict(int)
class_total = defaultdict(int)

for yt, yp in zip(y_true, y_pred):
    class_total[yt] += 1
    if yt == yp:
        class_correct[yt] += 1

df = pd.DataFrame([
    {
        "Class": class_names[i],
        "Accuracy": class_correct[i] / class_total[i]
    }
    for i in class_total
]).sort_values("Accuracy")

st.bar_chart(df.head(30).set_index("Class"))

st.header("Per-Class Strengths")
st.bar_chart(df.tail(30).set_index("Class"))


#-------Confusion Pairs (table)------------------------------------------------
st.header("Most Common Confusions")

confusions = defaultdict(int)

for yt, yp in zip(y_true, y_pred):
    if yt != yp:
        confusions[(class_names[yt], class_names[yp])] += 1

conf_df = pd.DataFrame(
    [(k[0], k[1], v) for k, v in confusions.items()],
    columns=["True", "Predicted", "Count"]
).sort_values("Count", ascending=False)

st.dataframe(conf_df.head(15))

#-------Wrong Predicions with high confidence gallery------------------------------
st.header("Gallery: Some Wrong Predicions with high confidence")

data = np.load(
    "/mnt/d/04_Food101-EfficientNetV2S-model/artifacts/selected_misclassified.npz",
    allow_pickle=True
)
indices = data["indices"]
images = data["images"]
true_labels = data["labels"]

import matplotlib.pyplot as plt

confidence = np.max(y_probs, axis=1)

fig = plt.figure(figsize=(15, 10))

for j in range(len(indices)):
    idx = indices[j]
    image = images[j]
    true_label = true_labels[j]

    pred_label = y_pred[idx]
    conf = confidence[idx]

    ax = fig.add_subplot(3, 4, j + 1)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"True: {class_names[true_label]}\n"
        f"Pred: {class_names[pred_label]} ({conf:.2f})",
        fontsize=9
    )
    
st.caption(
    "These are the model’s most confident mistakes, which are the most critical to analyze for real-world deployment."
)

plt.tight_layout()
st.pyplot(fig)                  