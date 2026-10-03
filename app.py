"""
Shoe Type Classification - Streamlit web app
Run with:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------- Basic settings ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "shoe_classifier.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"
IMG_SIZE = (224, 224)
DEFAULT_CLASSES = ["boots", "heels", "loafers", "sandals", "sneakers"]  # alphabetical order
EMOJI = {"sneakers": "👟", "boots": "🥾", "sandals": "🩴", "heels": "👠", "loafers": "👞"}

st.set_page_config(page_title="Shoe Type Classifier", page_icon="👟")


# ---------- Load model once (cached) ----------
@st.cache_resource
def load_model_and_classes():
    import tensorflow as tf  # imported here so the page opens quickly

    model = tf.keras.models.load_model(MODEL_PATH)
    if CLASS_PATH.exists():
        class_names = json.loads(CLASS_PATH.read_text())
    else:
        class_names = DEFAULT_CLASSES
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 and add a batch dimension.
    (Pixel scaling for MobileNetV2 is already inside the saved model.)"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype="float32")
    return np.expand_dims(arr, axis=0)


# ---------- Page ----------
st.title("👟 Shoe Type Classification")
st.write("Upload a shoe photo and the AI model will tell you what type of shoe it is.")
st.caption("Classes: Sneakers • Boots • Sandals • Heels • Loafers")

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: `model/shoe_classifier.keras`.\n\n"
        "Please run all cells in **train_model.ipynb** first to train and save the model."
    )
    st.stop()

with st.spinner("Loading model..."):
    model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Step 1: Upload a shoe image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=320)

    if st.button("Step 2: Predict", type="primary"):
        with st.spinner("Thinking..."):
            preds = model.predict(preprocess(image), verbose=0)[0]

        best = int(np.argmax(preds))
        label = class_names[best]
        confidence = float(preds[best]) * 100

        st.success(f"{EMOJI.get(label, '')} Predicted shoe type: **{label.capitalize()}**")
        st.metric("Confidence", f"{confidence:.2f}%")

        if confidence < 50:
            st.warning("The model is not very sure. Try a clearer photo with a plain background.")

        st.subheader("All class probabilities")
        for i in np.argsort(preds)[::-1]:
            st.write(f"{class_names[i].capitalize()} — {preds[i] * 100:.2f}%")
            st.progress(float(preds[i]))
else:
    st.info("Please upload an image to begin.")
