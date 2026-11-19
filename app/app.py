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

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dataset import eval_tf, CLASSES
from model import build_resnet18
from gradcam import GradCAM
from leaf_detector import load_leaf_detector
from report_pdf import generate_pdf_report

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="CropShield — AI Plant Health & Advisory",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, real-life agritech feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.92;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Stat cards on the home page */
    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 20px 18px;
        border: 1px solid #e0e8e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.12);
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .stat-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #616161;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Clean upload box */
    .upload-card {
        background: #fbfdfb;
        border: 2px dashed #81c784;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 20px;
    }

    /* Diagnosis badge */
    .diagnosis-badge {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 12px;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #757575;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #eeeeee;
    }
</style>
""", unsafe_allow_html=True)

UI = {
    "English": {
        "title": "CropShield AI",
        "tagline": "Next-Gen Crop Disease Detection, Leaf Verification & Precision Treatment",
        "nav_diagnose": "🩺 Leaf Diagnosis",
        "nav_crops": "🌱 Supported Crops (38 Diseases)",
        "nav_about": "📊 Model Intelligence & Accuracy",
        "nav_kvk": "🏛️ KVK & Helplines",
        "upload_opt": "📁 Upload Photo",
        "camera_opt": "📷 Take Photo with Camera",
        "upload_label": "Upload a clear crop leaf photo (JPG, PNG)",
        "camera_label": "Take a photo of the crop leaf",
        "your_photo": "Submitted Leaf",
        "heatmap": "Model Attention (Grad-CAM)",
        "healthy_ref": "Certified Healthy Reference",
        "top3": "Top-3 Disease Probabilities",
        "not_a_leaf": "🚫 Rejected: Not a plant leaf ({leaf_prob:.0%} leaf probability). CropShield requires a clear photo of a crop leaf.",
        "not_sure": "⚠️ Uncertain diagnosis ({conf:.0%} confidence). Photo does not clearly match supported diseases. Please capture a closer, well-lit photo.",
        "conf_label": "Confidence",
        "type_label": "Pathogen Class",
        "healthy_msg": "✅ Healthy leaf! No visible pathogen infection detected.",
        "cause": "Pathogen Cause & Spread",
        "symptoms": "Visible Leaf Symptoms",
        "treatment": "Recommended Remedial Actions",
        "prevention": "Preventive Cultural Practices",
        "download_pdf": "📄 Download Official Advisory Report (PDF)",
        "select_state": "Select State / Region:",
        "ref_heading": "🔍 Visual Check: Healthy vs. Diseased Reference",
        "disclaimer": "⚠️ Disclaimer: CropShield is an AI-assisted agricultural decision support tool. Recommendations mention functional chemical groups only, never exact commercial dosages. Consult your local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying.",
    },
    "हिन्दी (Hindi)": {
        "title": "क्रॉपशील्ड (CropShield AI)",
        "tagline": "उन्नत फसल रोग पहचान, पत्ती सत्यापन एवं सटीक उपचार परामर्श",
        "nav_diagnose": "🩺 पत्ती जाँच व निदान",
        "nav_crops": "🌱 समर्थित फसलें (38 रोग)",
        "nav_about": "📊 मॉडल शुद्धता व रिपोर्ट",
        "nav_kvk": "🏛️ कृषि विज्ञान केंद्र (KVK)",
        "upload_opt": "📁 फोटो अपलोड करें",
        "camera_opt": "📷 कैमरे से फोटो लें",
        "upload_label": "पत्ती का फोटो चुनें (JPG, PNG)",
        "camera_label": "पत्ती का फोटो खींचें",
        "your_photo": "जाँच हेतु पत्ती",
        "heatmap": "मॉडल का ध्यान (Grad-CAM हीटमैप)",
        "healthy_ref": "स्वस्थ पत्ती का संदर्भ",
        "top3": "शीर्ष-3 संभावित रोग (Top-3)",
        "not_a_leaf": "🚫 अस्वीकृत: यह पौधे की पत्ती नहीं लग रही है ({leaf_prob:.0%} संभावना)। कृपया फसल की पत्ती का स्पष्ट फोटो लें।",
        "not_sure": "⚠️ अनिश्चित परिणाम ({conf:.0%} विश्वास)। फोटो किसी ज्ञात रोग से स्पष्ट रूप से मेल नहीं खाता। बेहतर रोशनी में स्पष्ट फोटो लें।",
        "conf_label": "विश्वसनीयता स्तर",
        "type_label": "रोगज़नक़ का प्रकार",
        "healthy_msg": "✅ स्वस्थ पत्ती! किसी भी रोगज़नक़ के लक्षण नहीं पाए गए।",
        "cause": "रोग का कारण एवं फैलाव",
        "symptoms": "दिखाई देने वाले लक्षण",
        "treatment": "उपचार के प्रमुख कदम (क्या करें)",
        "prevention": "रोकथाम व बचाव के उपाय",
        "download_pdf": "📄 आधिकारिक परामर्श रिपोर्ट डाउनलोड करें (PDF)",
        "select_state": "अपना राज्य चुनें:",
        "ref_heading": "🔍 स्वस्थ पत्ती और रोग की तुलना",
        "disclaimer": "⚠️ अस्वीकरण: CropShield एक एआई कृषि सहायता प्रणाली है। किसी भी रासायनिक छिड़काव से पहले अपने नजदीकी कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से पुष्टि करें।",
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

# Sidebar: Navigation & Language
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=60)
    st.markdown("### 🌿 CropShield Portal")
    
    # Language Toggle
    lang = st.radio("Language / भाषा", ["English", "हिन्दी (Hindi)"], horizontal=True)
    t = UI[lang]
    remedies = all_remedies[lang]
    
    st.markdown("---")
    # Multi-tab Navigation
    page = st.radio(
        "Navigation",
        [t["nav_diagnose"], t["nav_crops"], t["nav_about"], t["nav_kvk"]],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style="background:#e8f5e9; padding:12px; border-radius:10px; font-size:0.85rem; color:#1b5e20;">
            <b>🌾 Kisan Call Center:</b><br/>
            📞 <b>1800-180-1551</b> (Toll-Free)<br/>
            <i>Available 6:00 AM – 10:00 PM</i>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("SIH Problem Statement SIH26131 • v2.1")

# Top Hero Banner across all tabs
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">{t['title']}</div>
    <div class="hero-sub">{t['tagline']}</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: DIAGNOSIS & TREATMENT (CORE)
# ==========================================
if page == t["nav_diagnose"]:
    # Stat strip
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="stat-card"><div class="stat-val">38</div><div class="stat-lbl">Crop Diseases</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-card"><div class="stat-val">99.05%</div><div class="stat-lbl">Leaf Gatekeeper</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat-card"><div class="stat-val">87.39%</div><div class="stat-lbl">Field Top-3 Acc</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-card"><div class="stat-val">14</div><div class="stat-lbl">Major Crops</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Input selection
    mode = st.radio("Select Input Mode", [t["upload_opt"], t["camera_opt"]], horizontal=True, label_visibility="collapsed")
    
    col_up, col_info = st.columns([1.4, 1])
    with col_up:
        if mode == t["upload_opt"]:
            file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
        else:
            file = st.camera_input(t["camera_label"])
            
    with col_info:
        st.markdown("""
        <div style="background: #f4fbf4; border: 1px solid #c8e6c9; border-radius: 12px; padding: 16px; font-size: 0.9rem;">
            <b>📸 Tips for Best Accuracy:</b><br/>
            • Take a close-up photo of a single diseased leaf.<br/>
            • Ensure adequate natural daylight and avoid deep shadows.<br/>
            • Our <b>LeafDetector</b> automatically screens out selfies or non-leaf objects.
        </div>
        """, unsafe_allow_html=True)

    if file:
        img = Image.open(file).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        is_leaf, leaf_prob = detector.predict(x)

        if not is_leaf:
            st.error(t["not_a_leaf"].format(leaf_prob=leaf_prob))
            c_err, _ = st.columns([1, 2])
            c_err.image(img.resize((240, 240)), caption=t["your_photo"])
        else:
            heat, idx, conf, probs = gradcam(x)
            raw_class = CLASSES[idx]
            info = remedies[raw_class]
            crop_name = raw_class.split("___")[0]

            if conf < THRESHOLD:
                st.warning(t["not_sure"].format(conf=conf))
                c_warn, _ = st.columns([1, 2])
                c_warn.image(img.resize((240, 240)), caption=t["your_photo"])
            else:
                st.markdown("---")
                
                # Active Diagnosis Banner
                st.markdown(f"""
                <div class="diagnosis-badge">
                    🌾 {info['name']}
                </div>
                """, unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1.1, 1.4])
                
                with res_col1:
                    m1, m2 = st.columns(2)
                    m1.metric(t["conf_label"], f"{conf:.1%}")
                    m2.metric(t["type_label"], info["type"].title())

                    if raw_class.endswith("___healthy"):
                        st.success(t["healthy_msg"])

                    # Images: Input and Grad-CAM side by side
                    img_col1, img_col2 = st.columns(2)
                    img_col1.image(img.resize((224, 224)), caption=t["your_photo"], use_container_width=True)
                    heatmap_overlay_img = Image.fromarray((overlay(img, heat) * 255).astype(np.uint8))
                    img_col2.image(heatmap_overlay_img, caption=t["heatmap"], use_container_width=True)

                    # Top-3 predictions
                    st.markdown(f"**{t['top3']}**")
                    top3_idx = np.argsort(probs)[::-1][:3]
                    top3_list = []
                    for rank, i in enumerate(top3_idx, start=1):
                        p = float(probs[i])
                        cls_name = remedies[CLASSES[i]]["name"]
                        top3_list.append((cls_name, p))
                        st.progress(p, text=f"{rank}. {cls_name} — {p:.1%}")

                    # PDF Report Button
                    try:
                        pdf_bytes = generate_pdf_report(
                            user_img=img,
                            heatmap_img=heatmap_overlay_img,
                            info=info,
                            conf=conf,
                            top3_list=top3_list,
                            kvk_info=kvk_directory.get("All India (General)")
                        )
                        st.download_button(
                            label=t["download_pdf"],
                            data=pdf_bytes,
                            file_name=f"CropShield_{raw_class}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.caption(f"PDF generator notice: {e}")

                with res_col2:
                    tab_act, tab_sym, tab_prev = st.tabs([
                        f"💊 {t['treatment']}",
                        f"🔬 {t['symptoms']}",
                        f"🛡️ {t['prevention']}"
                    ])

                    with tab_act:
                        st.markdown(f"### {t['treatment']}")
                        for s in info["treatment"]:
                            st.markdown(f"• **{s}**")

                    with tab_sym:
                        st.markdown(f"### {t['cause']}")
                        st.write(info["cause"])
                        st.markdown(f"### {t['symptoms']}")
                        st.write(info["symptoms"])

                    with tab_prev:
                        st.markdown(f"### {t['prevention']}")
                        for s in info["prevention"]:
                            st.markdown(f"• **{s}**")

                    # Reference check
                    healthy_ref_img = get_healthy_reference_image(crop_name)
                    if healthy_ref_img and not raw_class.endswith("___healthy"):
                        with st.expander(t["ref_heading"], expanded=False):
                            c_r1, c_r2 = st.columns(2)
                            c_r1.image(healthy_ref_img.resize((200, 200)), caption=f"Healthy {crop_name.replace('_', ' ')}")
                            c_r2.image(img.resize((200, 200)), caption=f"Diagnosed Leaf")

                st.caption(t["disclaimer"])


# ==========================================
# PAGE 2: SUPPORTED CROPS DIRECTORY
# ==========================================
elif page == t["nav_crops"]:
    st.subheader("🌱 Supported Crops & Disease Library (38 Classes)")
    st.write("Browse through all 14 crops and 38 conditions covered by CropShield's diagnostic model:")

    crops = sorted(list(set(c.split("___")[0] for c in CLASSES)))
    chosen_crop = st.selectbox("Select Crop to View Conditions:", crops)

    crop_classes = [c for c in CLASSES if c.startswith(chosen_crop + "___")]
    
    cols = st.columns(len(crop_classes) if len(crop_classes) <= 3 else 3)
    for idx, c in enumerate(crop_classes):
        with cols[idx % 3]:
            info = remedies[c]
            st.markdown(f"""
            <div style="border: 1px solid #c8e6c9; border-radius: 10px; padding: 12px; margin-bottom: 12px; background: white;">
                <h4 style="color: #1b5e20; margin-bottom: 6px;">{info['name']}</h4>
                <b>Type:</b> {info['type'].title()}<br/>
                <p style="font-size: 0.85rem; color: #555; margin-top: 4px;">{info['symptoms'][:120]}...</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: MODEL ACCURACY & EXPERIMENTAL REPORTS
# ==========================================
elif page == t["nav_about"]:
    st.subheader("📊 Experimental Results & Model Architecture")
    st.write("Verified metrics across studio benchmark (PlantVillage) and real-world field validation (PlantDoc):")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔬 Studio Benchmark (PlantVillage — 54,305 Images)")
        st.dataframe({
            "Model": ["SimpleCNN (Baseline)", "ResNet18 (Pretrained)"],
            "Parameters": ["399,142", "11,196,006"],
            "Test Accuracy": ["95.47%", "99.69%"],
            "Test Mistakes": ["246", "17"],
            "Macro F1": ["0.942", "0.996"]
        }, hide_index=True)

    with c2:
        st.markdown("#### 🌾 Real Field Test & Adaptation (PlantDoc — 2,575 Images)")
        st.dataframe({
            "Stage": ["ResNet18 Before Fine-Tuning", "ResNet18 After Field Augmentation"],
            "PlantDoc Top-1 Acc": ["20.17%", "61.34% (+41.17 pp)"],
            "PlantDoc Top-3 Acc": ["43.28%", "87.39% (+44.11 pp)"],
            "PlantVillage Top-1": ["99.69%", "98.47% (Preserved)"],
        }, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🛡️ LeafDetector (Out-of-Distribution Rejection)")
    st.write("Ensures non-plant photos (faces, objects, buildings) are screened out with **99.05% binary accuracy** and **0.9998 ROC-AUC** before disease classification runs.")


# ==========================================
# PAGE 4: KVK DIRECTORY & EMERGENCY HELPLINES
# ==========================================
elif page == t["nav_kvk"]:
    st.subheader("🏛️ Krishi Vigyan Kendra (KVK) Directory & Farm Helplines")
    st.write("Find direct agricultural contact details for your region:")

    st.markdown(f"#### {t['select_state']}")
    state_choice = st.selectbox("State", list(kvk_directory.keys()), label_visibility="collapsed")
    kvk_selected = kvk_directory[state_choice]

    col_kvk_card, _ = st.columns([2, 1])
    with col_kvk_card:
        st.markdown(f"""
        <div style="background: white; border: 2px solid #a5d6a7; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <h3 style="color: #1b5e20; margin-top: 0;">{kvk_selected['kvk']}</h3>
            <p><b>📍 Districts Covered:</b> {kvk_selected['district']}</p>
            <p><b>📞 Contact:</b> {kvk_selected['contact']}</p>
            <p style="background: #e8f5e9; padding: 10px; border-radius: 8px;">
                <b>🌾 Kisan Call Center:</b> <span style="color:#b71c1c; font-size:1.1rem; font-weight:700;">{kvk_selected['helpline']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# Shared Footer
st.markdown("""
<div class="app-footer">
    CropShield — Developed for Smart India Hackathon (SIH26131) • Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
