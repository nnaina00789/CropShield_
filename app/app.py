import sys
import json
from pathlib import Path
import numpy as np
import torch
import streamlit as st
import matplotlib.cm as cm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # so we can import from src/
from dataset import eval_tf, CLASSES
from model import build_resnet18
from gradcam import GradCAM

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource  # load the model and remedies once, not on every click
def load():
    model = build_resnet18().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/resnet18.pt", map_location=DEVICE))
    remedies = json.load(open("app/remedies.json", encoding="utf-8"))
    return GradCAM(model, model.layer4), remedies

def overlay(img, heat):
    colored = cm.jet(heat)[..., :3]                      # heat values 0..1 -> RGB colours
    base = np.asarray(img.resize((224, 224))) / 255.0
    return 0.55 * base + 0.45 * colored                  # blend photo and heatmap

st.set_page_config(page_title="CropShield", page_icon="🌿")
st.title("🌿 CropShield")
st.write("Upload a photo of a single leaf to detect disease, see where the model looked, and get treatment guidance.")
st.caption("Supported crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, "
           "potato, raspberry, soybean, squash, strawberry, tomato.")

gradcam, remedies = load()
file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    heat, idx, conf = gradcam(eval_tf(img).unsqueeze(0).to(DEVICE))
    info = remedies[CLASSES[idx]]

    col1, col2 = st.columns(2)
    col1.image(img.resize((224, 224)), caption="Your photo")
    col2.image(overlay(img, heat), caption="Where the model looked (red = most important)", clamp=True)

    if conf < THRESHOLD:
        st.warning(f"Not sure ({conf:.0%} confidence). Best guess: {info['name']}. "
                   "Try a clearer photo of a single leaf on a plain background, or consult an expert.")
    else:
        st.subheader(info["name"])
        st.write(f"**Confidence:** {conf:.0%}  ·  **Type:** {info['type']}")
        if info["type"] == "healthy":
            st.success("No disease detected.")
        st.write(f"**Cause:** {info['cause']}")
        st.write(f"**Symptoms:** {info['symptoms']}")
        st.write("**What to do:**")
        for step in info["treatment"]:
            st.write(f"- {step}")
        st.write("**Prevention:**")
        for step in info["prevention"]:
            st.write(f"- {step}")

    st.caption("CropShield gives general guidance only. Confirm with your local agriculture officer or "
               "Krishi Vigyan Kendra (KVK) before applying any treatment, and always follow the product label.")