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
from leaf_detector import load_leaf_detector

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UI = {
    "English": {
        "subtitle": "Upload or take a photo of a single leaf to detect disease, see where the model looked, and get treatment guidance.",
        "crops": ("Supported crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, "
                  "potato, raspberry, soybean, squash, strawberry, tomato."),
        "input_mode": "Photo source",
        "upload_opt": "Upload photo",
        "camera_opt": "Take photo with camera",
        "upload_label": "Leaf photo",
        "camera_label": "Take a leaf photo",
        "your_photo": "Your photo",
        "heatmap": "Where the model looked (red = most important)",
        "top3": "Top-3 predictions",
        "not_a_leaf": ("Not a plant leaf ({leaf_prob:.0%} leaf probability). "
                       "Please upload or take a clear photo of a plant leaf."),
        "not_sure": ("Not sure ({conf:.0%} confidence) — this does not clearly match a supported crop leaf. "
                     "Try a clearer photo of a single leaf on a plain background, or consult an expert."),
        "conf_type": "**Confidence:** {conf:.0%}  ·  **Type:** {type}",
        "healthy_msg": "No disease detected.",
        "cause": "**Cause:**",
        "symptoms": "**Symptoms:**",
        "treatment": "**What to do:**",
        "prevention": "**Prevention:**",
        "disclaimer": ("CropShield gives general guidance only. Confirm with your local agriculture officer or "
                       "Krishi Vigyan Kendra (KVK) before applying any treatment, and always follow the product label."),
    },
    "हिन्दी (Hindi)": {
        "subtitle": "रोग की पहचान करने, मॉडल ने कहाँ देखा यह जानने और उपचार संबंधी सलाह पाने के लिए एक पत्ती का फोटो अपलोड करें या कैमरे से लें।",
        "crops": ("समर्थित फसलें: सेब, ब्लूबेरी, चेरी, मक्का, अंगूर, संतरा, आड़ू, शिमला मिर्च, "
                  "आलू, रास्पबेरी, सोयाबीन, स्क्वैश (कद्दू वर्गीय), स्ट्रॉबेरी, टमाटर।"),
        "input_mode": "फोटो का स्रोत",
        "upload_opt": "फोटो अपलोड करें",
        "camera_opt": "कैमरे से फोटो लें",
        "upload_label": "पत्ती का फोटो चुनें",
        "camera_label": "पत्ती का फोटो लें",
        "your_photo": "आपका फोटो",
        "heatmap": "मॉडल ने कहाँ देखा (लाल = सबसे महत्वपूर्ण)",
        "top3": "शीर्ष-3 अनुमान (Top-3)",
        "not_a_leaf": ("यह पौधे की पत्ती नहीं लग रही है ({leaf_prob:.0%} पत्ती संभावना)। "
                       "कृपया किसी पौधे की पत्ती का स्पष्ट फोटो अपलोड करें या कैमरे से लें।"),
        "not_sure": ("निश्चित नहीं ({conf:.0%} विश्वास) — यह किसी समर्थित फसल की पत्ती से स्पष्ट रूप से मेल नहीं खाता। "
                     "सादे पृष्ठभूमि पर एक पत्ती का स्पष्ट फोटो लेने का प्रयास करें, या किसी कृषि विशेषज्ञ से सलाह लें।"),
        "conf_type": "**विश्वास (Confidence):** {conf:.0%}  ·  **प्रकार:** {type}",
        "healthy_msg": "कोई रोग नहीं पाया गया।",
        "cause": "**कारण:**",
        "symptoms": "**लक्षण:**",
        "treatment": "**उपचार (क्या करें):**",
        "prevention": "**रोकथाम (बचाव):**",
        "disclaimer": ("CropShield केवल सामान्य मार्गदर्शन देता है। किसी भी उपचार का प्रयोग करने से पहले अपने स्थानीय कृषि अधिकारी या "
                       "कृषि विज्ञान केंद्र (KVK) से पुष्टि करें, और हमेशा उत्पाद लेबल पर दिए गए निर्देशों का पालन करें।"),
    },
}

@st.cache_resource  # load the models and remedies once, not on every click
def load():
    detector = load_leaf_detector(device=DEVICE)
    model = build_resnet18().to(DEVICE)
    ckpt = "checkpoints/resnet18_field.pt" if Path("checkpoints/resnet18_field.pt").exists() else "checkpoints/resnet18.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    remedies_en = json.load(open("app/remedies.json", encoding="utf-8"))
    remedies_hi = json.load(open("app/remedies_hi.json", encoding="utf-8"))
    return detector, GradCAM(model, model.layer4), {"English": remedies_en, "हिन्दी (Hindi)": remedies_hi}

def overlay(img, heat):
    colored = cm.jet(heat)[..., :3]                      # heat values 0..1 -> RGB colours
    base = np.asarray(img.resize((224, 224))) / 255.0
    return 0.55 * base + 0.45 * colored                  # blend photo and heatmap

st.set_page_config(page_title="CropShield", page_icon="🌿")

col_title, col_lang = st.columns([3, 1])
col_title.title("🌿 CropShield")
lang = col_lang.radio("Language / भाषा", ["English", "हिन्दी (Hindi)"], horizontal=True)
t = UI[lang]

st.write(t["subtitle"])
st.caption(t["crops"])

detector, gradcam, all_remedies = load()
remedies = all_remedies[lang]

mode = st.radio(t["input_mode"], [t["upload_opt"], t["camera_opt"]], horizontal=True)
if mode == t["upload_opt"]:
    file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input(t["camera_label"])

if file:
    img = Image.open(file).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)
    is_leaf, leaf_prob = detector.predict(x)

    col1, col2 = st.columns(2)
    col1.image(img.resize((224, 224)), caption=t["your_photo"])

    if not is_leaf:
        st.error(t["not_a_leaf"].format(leaf_prob=leaf_prob))
    else:
        heat, idx, conf, probs = gradcam(x)
        info = remedies[CLASSES[idx]]

        if conf < THRESHOLD:
            st.warning(t["not_sure"].format(conf=conf))
        else:
            col2.image(overlay(img, heat), caption=t["heatmap"], clamp=True)

            st.subheader(t["top3"])
            top3_idx = np.argsort(probs)[::-1][:3]
            for rank, i in enumerate(top3_idx, start=1):
                p = float(probs[i])
                cls_name = remedies[CLASSES[i]]["name"]
                st.progress(p, text=f"{rank}. {cls_name} — {p:.1%}")

            st.subheader(info["name"])
            st.write(t["conf_type"].format(conf=conf, type=info["type"]))
            if CLASSES[idx].endswith("___healthy"):
                st.success(t["healthy_msg"])
            st.write(f"{t['cause']} {info['cause']}")
            st.write(f"{t['symptoms']} {info['symptoms']}")
            st.write(t["treatment"])
            for step in info["treatment"]:
                st.write(f"- {step}")
            st.write(t["prevention"])
            for step in info["prevention"]:
                st.write(f"- {step}")

    st.caption(t["disclaimer"])
