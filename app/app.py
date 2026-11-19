import sys
import json
import io
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import streamlit as st
import matplotlib.cm as cm
from PIL import Image
from gtts import gTTS

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dataset import eval_tf, CLASSES
from model import build_resnet18
from gradcam import GradCAM
from leaf_detector import load_leaf_detector
from report_pdf import generate_pdf_report

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="CropShield — Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern, clean, non-congested UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8faf8;
        border-radius: 10px;
        border: 1px solid #e0e6e0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stProgress > div > div > div > div {
        background-color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

UI = {
    "English": {
        "title": "🌿 CropShield",
        "tagline": "AI-Powered Crop Disease Diagnosis, Leaf Validation & Field Advisory",
        "crops_supported": "Supported crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.",
        "input_mode": "Photo Source",
        "upload_opt": "📁 Upload Photo",
        "camera_opt": "📷 Take Photo with Camera",
        "upload_label": "Choose a leaf photo (JPG, PNG)",
        "camera_label": "Capture leaf photo",
        "your_photo": "Submitted Photo",
        "heatmap": "Model Attention (Grad-CAM)",
        "healthy_ref": "Standard Healthy Reference",
        "top3": "Top-3 Disease Candidates",
        "not_a_leaf": "🚫 Not a plant leaf ({leaf_prob:.0%} leaf probability). Please upload or capture a clear photo of a plant leaf.",
        "not_sure": "⚠️ Uncertain diagnosis ({conf:.0%} confidence). Photo does not clearly match known diseases. Try a clearer close-up or consult an expert.",
        "conf_label": "Confidence",
        "type_label": "Pathogen Type",
        "healthy_msg": "✅ Healthy leaf. No visible disease pathogens detected.",
        "cause": "Cause & Biology",
        "symptoms": "Visible Symptoms",
        "treatment": "Recommended Treatment Actions",
        "prevention": "Preventive Cultural Practices",
        "listen_audio": "🔊 Listen to Advisory",
        "generating_audio": "Generating voice advisory...",
        "download_pdf": "📄 Download Official Advisory Report (PDF)",
        "generating_pdf": "Preparing PDF report...",
        "kvk_heading": "🏛️ Nearest Krishi Vigyan Kendra (KVK) & Helplines",
        "select_state": "Select your State / Region:",
        "ref_heading": "🔍 Healthy vs. Disease Reference Comparison",
        "disclaimer": "⚠️ Disclaimer: CropShield is an AI-assisted decision-support system. Recommendations indicate general chemical classes only, never exact dosages. Always verify with your local agricultural officer or KVK before application.",
        "severity_mild": "Mild (<15% symptomatic area)",
        "severity_mod": "Moderate (15% - 40% symptomatic area)",
        "severity_sev": "Severe (>40% symptomatic area)",
        "severity_label": "Estimated Symptom Extent"
    },
    "हिन्दी (Hindi)": {
        "title": "🌿 क्रॉपशील्ड (CropShield)",
        "tagline": "एआई-आधारित फसल रोग पहचान, पत्ती सत्यापन एवं कृषि परामर्श",
        "crops_supported": "समर्थित फसलें: सेब, ब्लूबेरी, चेरी, मक्का, अंगूर, संतरा, आड़ू, शिमला मिर्च, आलू, रास्पबेरी, सोयाबीन, कद्दू वर्गीय (स्क्वैश), स्ट्रॉबेरी, टमाटर।",
        "input_mode": "फोटो का माध्यम",
        "upload_opt": "📁 फोटो अपलोड करें",
        "camera_opt": "📷 कैमरे से फोटो लें",
        "upload_label": "पत्ती का फोटो चुनें (JPG, PNG)",
        "camera_label": "पत्ती का फोटो खींचें",
        "your_photo": "आपका फोटो",
        "heatmap": "मॉडल का ध्यान (Grad-CAM हीटमैप)",
        "healthy_ref": "स्वस्थ पत्ती का संदर्भ",
        "top3": "शीर्ष-3 संभावित रोग (Top-3)",
        "not_a_leaf": "🚫 यह किसी पौधे की पत्ती नहीं लग रही है ({leaf_prob:.0%} संभावना)। कृपया किसी पौधे की पत्ती का स्पष्ट फोटो लें।",
        "not_sure": "⚠️ अनिश्चित परिणाम ({conf:.0%} विश्वास)। फोटो किसी ज्ञात रोग से स्पष्ट रूप से मेल नहीं खाता। सादे बैकग्राउंड पर स्पष्ट फोटो लें या विशेषज्ञ से मिलें।",
        "conf_label": "विश्वास स्तर (Confidence)",
        "type_label": "रोगज़नक़ का प्रकार",
        "healthy_msg": "✅ पत्ती पूर्णतः स्वस्थ है। किसी रोग के लक्षण नहीं पाए गए।",
        "cause": "रोग का कारण एवं जीवविज्ञान",
        "symptoms": "दिखाई देने वाले लक्षण",
        "treatment": "उपचार के प्रमुख कदम (क्या करें)",
        "prevention": "भविष्य में रोकथाम व बचाव",
        "listen_audio": "🔊 परामर्श सुनें (आवाज में)",
        "generating_audio": "ऑडियो तैयार हो रहा है...",
        "download_pdf": "📄 आधिकारिक परामर्श रिपोर्ट डाउनलोड करें (PDF)",
        "generating_pdf": "पीडीएफ रिपोर्ट तैयार हो रही है...",
        "kvk_heading": "🏛️ नजदीकी कृषि विज्ञान केंद्र (KVK) व हेल्पलाइन",
        "select_state": "अपना राज्य चुनें:",
        "ref_heading": "🔍 स्वस्थ पत्ती और रोग की तुलना",
        "disclaimer": "⚠️ अस्वीकरण: CropShield केवल सामान्य मार्गदर्शन प्रदान करता है। किसी भी रासायनिक छिड़काव से पहले स्थानीय कृषि अधिकारी या नजदीकी KVK से पुष्टि करें।",
        "severity_mild": "हल्का संक्रमण (<15% पत्ती क्षेत्र)",
        "severity_mod": "मध्यम संक्रमण (15% - 40% पत्ती क्षेत्र)",
        "severity_sev": "गंभीर संक्रमण (>40% पत्ती क्षेत्र)",
        "severity_label": "अनुमानित संक्रमण स्तर"
    }
}

@st.cache_resource
def load_assets():
    detector = load_leaf_detector(device=DEVICE)
    model = build_resnet18().to(DEVICE)
    ckpt = "checkpoints/resnet18_field.pt" if Path("checkpoints/resnet18_field.pt").exists() else "checkpoints/resnet18.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    remedies_en = json.load(open("app/remedies.json", encoding="utf-8"))
    remedies_hi = json.load(open("app/remedies_hi.json", encoding="utf-8"))
    kvk_data = json.load(open("app/kvk_directory.json", encoding="utf-8"))
    return detector, GradCAM(model, model.layer4), {"English": remedies_en, "हिन्दी (Hindi)": remedies_hi}, kvk_data

def overlay(img, heat):
    colored = cm.jet(heat)[..., :3]
    base = np.asarray(img.resize((224, 224))) / 255.0
    return 0.55 * base + 0.45 * colored

def get_healthy_reference_image(crop_name: str):
    healthy_key = f"{crop_name}___healthy"
    demo_path = Path("demo_images") / f"{healthy_key}.jpg"
    if demo_path.exists():
        return Image.open(demo_path)
    return None

detector, gradcam, all_remedies, kvk_directory = load_assets()

# Sidebar: Language Selection, About, and KVK Directory
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=64)
    st.markdown("### Language / भाषा")
    lang = st.radio("Choose Language", ["English", "हिन्दी (Hindi)"], label_visibility="collapsed")
    t = UI[lang]
    remedies = all_remedies[lang]

    st.markdown("---")
    st.markdown(f"#### {t['kvk_heading']}")
    state_list = list(kvk_directory.keys())
    selected_state = st.selectbox(t["select_state"], state_list, index=0)
    kvk_info = kvk_directory[selected_state]
    st.info(f"**Centre:** {kvk_info['kvk']}\n\n**Contact:** {kvk_info['contact']}\n\n**Helpline:** {kvk_info['helpline']}")

    st.markdown("---")
    st.caption("CropShield v2.0 • Field-adapted ResNet18 • LeafDetector OOD Gatekeeper")

# Main Page Header
col_header, _ = st.columns([4, 1])
with col_header:
    st.markdown(f"<div class='main-header'>{t['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{t['tagline']}</div>", unsafe_allow_html=True)

st.caption(t["crops_supported"])

# Input Mode Selector
mode = st.radio(t["input_mode"], [t["upload_opt"], t["camera_opt"]], horizontal=True, label_visibility="collapsed")

if mode == t["upload_opt"]:
    file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input(t["camera_label"])

if file:
    img = Image.open(file).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)
    is_leaf, leaf_prob = detector.predict(x)

    # 1. Leaf vs. Not-a-Leaf Screening
    if not is_leaf:
        st.error(t["not_a_leaf"].format(leaf_prob=leaf_prob))
        col_err, _ = st.columns([1, 2])
        col_err.image(img.resize((224, 224)), caption=t["your_photo"])
    else:
        heat, idx, conf, probs = gradcam(x)
        raw_class = CLASSES[idx]
        info = remedies[raw_class]
        crop_name = raw_class.split("___")[0]

        # 2. Confidence Threshold Check
        if conf < THRESHOLD:
            st.warning(t["not_sure"].format(conf=conf))
            col_warn, _ = st.columns([1, 2])
            col_warn.image(img.resize((224, 224)), caption=t["your_photo"])
        else:
            # Main Result Section
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.1, 1.4])

            with res_col1:
                st.subheader(info["name"])
                
                # Badges / Metrics
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(t["conf_label"], f"{conf:.1%}")
                m_col2.metric(t["type_label"], info["type"].title())

                if raw_class.endswith("___healthy"):
                    st.success(t["healthy_msg"])

                # Image pair: Original + Grad-CAM
                i_col1, i_col2 = st.columns(2)
                i_col1.image(img.resize((224, 224)), caption=t["your_photo"])
                heatmap_overlay_img = Image.fromarray((overlay(img, heat) * 255).astype(np.uint8))
                i_col2.image(heatmap_overlay_img, caption=t["heatmap"])

                # Top-3 Predictions Bar
                st.markdown(f"##### {t['top3']}")
                top3_idx = np.argsort(probs)[::-1][:3]
                top3_list = []
                for rank, i in enumerate(top3_idx, start=1):
                    p = float(probs[i])
                    cls_name = remedies[CLASSES[i]]["name"]
                    top3_list.append((cls_name, p))
                    st.progress(p, text=f"{rank}. {cls_name} — {p:.1%}")

                # Voice / Audio Advisory
                st.markdown("---")
                audio_lang = "hi" if lang == "हिन्दी (Hindi)" else "en"
                audio_text = f"{info['name']}. {info['cause']}. "
                if info.get("treatment"):
                    audio_text += f"{t['treatment']}: " + " ".join(info["treatment"][:2])

                try:
                    audio_buf = io.BytesIO()
                    gTTS(text=audio_text, lang=audio_lang).write_to_fp(audio_buf)
                    audio_buf.seek(0)
                    st.markdown(f"**{t['listen_audio']}**")
                    st.audio(audio_buf, format="audio/mp3")
                except Exception:
                    pass

                # PDF Download Button
                try:
                    pdf_bytes = generate_pdf_report(
                        user_img=img,
                        heatmap_img=heatmap_overlay_img,
                        info=info,
                        conf=conf,
                        top3_list=top3_list,
                        kvk_info=kvk_info
                    )
                    st.download_button(
                        label=t["download_pdf"],
                        data=pdf_bytes,
                        file_name=f"CropShield_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.caption(f"PDF export notice: {e}")

            with res_col2:
                # Tabbed Advice to prevent cluttered scrolling
                tab_treatment, tab_symptoms, tab_prevention = st.tabs([
                    f"💊 {t['treatment']}",
                    f"🔬 {t['symptoms']}",
                    f"🛡️ {t['prevention']}"
                ])

                with tab_treatment:
                    st.markdown(f"**{t['treatment']}**")
                    for step in info["treatment"]:
                        st.markdown(f"- {step}")

                with tab_symptoms:
                    st.markdown(f"**{t['cause']}**")
                    st.write(info["cause"])
                    st.markdown(f"**{t['symptoms']}**")
                    st.write(info["symptoms"])

                with tab_prevention:
                    st.markdown(f"**{t['prevention']}**")
                    for step in info["prevention"]:
                        st.markdown(f"- {step}")

                # Healthy vs. Diseased Reference Gallery
                healthy_ref_img = get_healthy_reference_image(crop_name)
                if healthy_ref_img and not raw_class.endswith("___healthy"):
                    with st.expander(t["ref_heading"], expanded=False):
                        st.caption(f"Compare your leaf against a certified healthy {crop_name.replace('_', ' ')} leaf:")
                        c_ref1, c_ref2 = st.columns(2)
                        c_ref1.image(healthy_ref_img.resize((224, 224)), caption=f"Healthy {crop_name.replace('_', ' ')}")
                        c_ref2.image(img.resize((224, 224)), caption=f"Your Leaf ({info['name']})")

            st.caption(t["disclaimer"])
