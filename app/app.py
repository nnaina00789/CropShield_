import sys
import json
import io
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.cm as cm
import pydeck as pdk
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
    page_title="CropShield — AI Plant Health & Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast, Professional Theme-Adaptive CSS
st.markdown("""
<style>
    /* Card Container */
    .cs-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Workflow Steps Card */
    .step-box {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        height: 100%;
    }
    .step-num {
        font-size: 1.1rem;
        font-weight: 800;
        color: #2e7d32;
        margin-bottom: 4px;
    }
    .step-text {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Hero Header */
    .cs-hero {
        padding: 10px 0 16px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }
    .cs-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .cs-tagline {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-top: 6px;
    }

    /* Region / Intro Card */
    .region-card {
        background-color: var(--secondary-background-color);
        border-top: 4px solid #2e7d32;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        border-left: 1px solid rgba(128, 128, 128, 0.2);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .region-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 6px;
    }

    /* Diagnosis Result Banner */
    .cs-result-banner {
        background-color: rgba(46, 125, 50, 0.12);
        border-left: 6px solid #2e7d32;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .cs-result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

UI = {
    "English": {
        "title": "🌿 CropShield AI",
        "tagline": "Empowering Indian Agriculture with Computer Vision & Targeted Crop Health Diagnostics",
        "nav_overview": "🌾 Indian Agriculture & Crops",
        "nav_diagnose": "🩺 Leaf Diagnosis",
        "nav_crops": "🌱 Supported Crops (38 Classes)",
        "nav_about": "📊 Benchmark Metrics & Architecture",
        "nav_kvk": "🏛️ KVK & Farmer Helplines",
        "step1_title": "1. 📸 Capture / Upload",
        "step1_desc": "Upload a crop leaf photo or capture one using your webcam / phone camera.",
        "step2_title": "2. 🛡️ AI Verification",
        "step2_desc": "LeafDetector validates the image, and Grad-CAM pinpoints the exact lesion.",
        "step3_title": "3. 💊 Treatment & Report",
        "step3_desc": "Get targeted disease guidance and download a print-ready advisory PDF.",
        "test_guide_title": "💡 How to test the system (Quick Guide):",
        "test_guide_1": "• **Test Crop Diseases:** Upload any sample from the `demo_images/` folder (Tomato, Potato, Corn, Grape, Apple, etc.).",
        "test_guide_2": "• **Test 'Not a Leaf' Rejection:** Upload a photo of a face, building, pet, or random object — the gatekeeper will reject it with zero false diagnosis.",
        "test_guide_3": "• **Field Robustness:** The classifier is fine-tuned on real in-situ farm photos (PlantDoc) to handle background soil and shadows.",
        "upload_opt": "📁 Upload Leaf Photo",
        "camera_opt": "📷 Capture Live Photo",
        "upload_label": "Upload a clear crop leaf photo (JPG, PNG)",
        "camera_label": "Capture photo with camera",
        "your_photo": "Submitted Leaf Photo",
        "heatmap": "Model Attention Map (Grad-CAM)",
        "top3": "Top-3 Diagnostic Probabilities",
        "not_a_leaf": "🚫 Validation Failed: Not a plant leaf ({leaf_prob:.0%} leaf probability). CropShield requires a clear crop leaf photo.",
        "not_sure": "⚠️ Low Confidence ({conf:.0%}). The image does not clearly match any supported crop disease. Please take a clearer, well-lit photo.",
        "conf_label": "Confidence Score",
        "type_label": "Pathogen Category",
        "healthy_msg": "✅ Healthy crop leaf. No visible pathogen infection detected.",
        "cause": "Biological Origin & Spread",
        "symptoms": "Visible Symptoms",
        "treatment": "Recommended Remedial Actions",
        "prevention": "Preventive Cultural Practices",
        "download_pdf": "📄 Download Official Advisory Report (PDF)",
        "select_state": "Select State / Region:",
        "ref_heading": "🔍 Visual Check: Healthy Reference vs. Submitted Leaf",
        "disclaimer": "⚠️ Disclaimer: CropShield is an AI decision-support system. Recommendations mention chemical functional classes only, never commercial dosages. Always verify with local agricultural officers or KVK before application.",
    },
    "हिन्दी (Hindi)": {
        "title": "🌿 क्रॉपशील्ड (CropShield AI)",
        "tagline": "भारतीय कृषि के लिए कंप्यूटर विज़न आधारित फसल रोग पहचान एवं उपचार प्रणाली",
        "nav_overview": "🌾 भारतीय कृषि एवं फसलें",
        "nav_diagnose": "🩺 पत्ती जाँच व निदान",
        "nav_crops": "🌱 समर्थित फसलें (38 रोग)",
        "nav_about": "📊 मॉडल शुद्धता व रिपोर्ट",
        "nav_kvk": "🏛️ कृषि विज्ञान केंद्र (KVK)",
        "step1_title": "1. 📸 फोटो चुनें / खींचें",
        "step1_desc": "फसल की पत्ती का फोटो अपलोड करें या कैमरे से लाइव फोटो लें।",
        "step2_title": "2. 🛡️ एआई सत्यापन",
        "step2_desc": "LeafDetector पत्ती की पुष्टि करता है और Grad-CAM रोग का क्षेत्र दिखाता है।",
        "step3_title": "3. 💊 उपचार व रिपोर्ट",
        "step3_desc": "सटीक उपचार के उपाय देखें और आधिकारिक पीडीएफ रिपोर्ट डाउनलोड करें।",
        "test_guide_title": "💡 सिस्टम की जाँच कैसे करें (त्वरित मार्गदर्शिका):",
        "test_guide_1": "• **फसल रोगों की जाँच:** `demo_images/` फ़ोल्डर से किसी भी पत्ती का फोटो अपलोड करें (टमाटर, आलू, मक्का, अंगूर, सेब आदि)।",
        "test_guide_2": "• **'पत्ती नहीं है' की जाँच:** किसी चेहरे, इमारत या वस्तु का फोटो अपलोड करें — मॉडल इसे तुरंत अस्वीकार कर देगा।",
        "test_guide_3": "• **खेत की परिस्थितियों में सटीकता:** मॉडल को खेत के वास्तविक वातावरण (मिट्टी, धूप, छाया) के लिए तैयार किया गया है।",
        "upload_opt": "📁 फोटो अपलोड करें",
        "camera_opt": "📷 कैमरे से फोटो लें",
        "upload_label": "पत्ती का फोटो चुनें (JPG, PNG)",
        "camera_label": "पत्ती का फोटो खींचें",
        "your_photo": "जाँच हेतु पत्ती का फोटो",
        "heatmap": "मॉडल का ध्यान (Grad-CAM हीटमैप)",
        "top3": "शीर्ष-3 संभावित रोग (Top-3)",
        "not_a_leaf": "🚫 सत्यापन विफल: यह पौधे की पत्ती नहीं है ({leaf_prob:.0%} संभावना)। कृपया फसल की पत्ती का स्पष्ट फोटो अपलोड करें।",
        "not_sure": "⚠️ कम विश्वास स्तर ({conf:.0%})। फोटो किसी ज्ञात फसल रोग से मेल नहीं खाती। कृपया बेहतर रोशनी में स्पष्ट फोटो लें।",
        "conf_label": "विश्वास स्तर",
        "type_label": "रोगज़नक़ का प्रकार",
        "healthy_msg": "✅ स्वस्थ पत्ती! किसी भी रोगज़नक़ के लक्षण नहीं पाए गए।",
        "cause": "रोग का कारण एवं फैलाव",
        "symptoms": "दिखाई देने वाले लक्षण",
        "treatment": "उपचार के प्रमुख कदम (क्या करें)",
        "prevention": "रोकथाम व बचाव के उपाय",
        "download_pdf": "📄 आधिकारिक परामर्श रिपोर्ट डाउनलोड करें (PDF)",
        "select_state": "अपना राज्य चुनें:",
        "ref_heading": "🔍 स्वस्थ पत्ती और रोग की तुलना",
        "disclaimer": "⚠️ अस्वीकरण: CropShield केवल सामान्य मार्गदर्शन प्रदान करता है। किसी भी रासायनिक छिड़काव से पहले नजदीकी कृषि विज्ञान केंद्र (KVK) से पुष्टि करें।",
    }
}

# Agro-climatic regions database for India
AGRO_REGIONS = {
    "🍎 Himalayan Temperate Zone (North)": {
        "states": "Himachal Pradesh, Jammu & Kashmir, Uttarakhand",
        "lat": 32.0, "lon": 76.5,
        "crops": ["Apple", "Peach", "Cherry"],
        "soil": "Brown hill soil, acidic mountain soil",
        "irrigation": "Natural mountain springs, rainfed contours",
        "climate": "Chilling winters, temperate summers (15°C–28°C)",
        "threats": "Apple Scab (Venturia), Cedar Apple Rust, Powdery Mildew",
        "why": "Spring snow-melt creates high ambient humidity, triggering fungal spore release."
    },
    "🌽 Indo-Gangetic Plains (North & Central)": {
        "states": "Uttar Pradesh, Punjab, Haryana, Bihar, Madhya Pradesh",
        "lat": 26.8, "lon": 80.9,
        "crops": ["Potato", "Corn_(maize)", "Soybean", "Tomato"],
        "soil": "Alluvial deep fertile loam (Khadar and Bhangar)",
        "irrigation": "Extensive canal networks, tube-wells",
        "climate": "Extreme summers (40°C+), cold humid winters (5°C–18°C)",
        "threats": "Late Blight (Phytophthora) in potato/tomato, Corn Leaf Blight, Early Blight",
        "why": "Heavy winter morning dew and dense fog create perfect incubation for Phytophthora water mould."
    },
    "🍇 Deccan Plateau & Western Semiarid Belt": {
        "states": "Maharashtra, Karnataka, Telangana, Gujarat",
        "lat": 19.5, "lon": 74.5,
        "crops": ["Grape", "Pepper,_bell", "Tomato", "Squash"],
        "soil": "Black cotton soil (Regur), volcanic basaltic loam",
        "irrigation": "Precision drip systems, farm ponds (Tals)",
        "climate": "Semi-arid, tropical wet-and-dry (20°C–36°C)",
        "threats": "Grape Black Rot, Esca (Black Measles), Bell Pepper Bacterial Spot",
        "why": "Sudden unseasonal post-monsoon showers cause rapid berry cracking and fungal entry."
    },
    "🍊 Subtropical & Coastal Horticulture Belts": {
        "states": "Punjab/Rajasthan (Citrus), Andhra Pradesh, Tamil Nadu, Mahabaleshwar",
        "lat": 15.5, "lon": 78.5,
        "crops": ["Orange", "Strawberry", "Blueberry", "Raspberry"],
        "soil": "Red sandy loam, laterite, enriched peat for berries",
        "irrigation": "Micro-sprinklers, raised beds with silver-black mulch",
        "climate": "Subtropical, frost-free winters, high sunshine hours",
        "threats": "Citrus Greening (Huanglongbing), Strawberry Leaf Scorch, Spider Mites",
        "why": "Sap-sucking psyllid insects and warm dry air accelerate vector-borne viral/bacterial spread."
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

# Sidebar: Language & Navigation
with st.sidebar:
    st.markdown("### 🌿 CropShield Navigation")
    
    lang = st.radio("Language / भाषा", ["English", "हिन्दी (Hindi)"], horizontal=True)
    t = UI[lang]
    remedies = all_remedies[lang]
    
    st.markdown("---")
    page = st.radio(
        "Select Portal Section",
        [t["nav_overview"], t["nav_diagnose"], t["nav_crops"], t["nav_about"], t["nav_kvk"]],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### 🌾 National Farmer Helpline")
    st.info("📞 **1800-180-1551**\n\nToll-Free Kisan Call Center • Available in 22 Regional Languages (6 AM – 10 PM)")

# Header Section
st.markdown(f"""
<div class="cs-hero">
    <div class="cs-title">{t['title']}</div>
    <div class="cs-tagline">{t['tagline']}</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 0: INTRO - INTERACTIVE INDIAN FARMING EXPLORER
# ==========================================
if page == t["nav_overview"]:
    st.subheader("🗺️ Interactive Agro-Climatic Explorer: Crops & Farming Systems of India")
    st.write(
        "India is characterized by distinct agricultural micro-climates. Explore how the **14 key crops supported by CropShield** "
        "are grown, irrigated, and protected across India's premier farming belts:"
    )

    # 1. Interactive Region Selector
    selected_reg_name = st.selectbox(
        "📍 Select an Agricultural Belt to Inspect:",
        list(AGRO_REGIONS.keys()),
        index=1
    )
    reg_data = AGRO_REGIONS[selected_reg_name]

    # 2. Interactive Map & Deep-Dive Info Card Side by Side
    col_map, col_details = st.columns([1.1, 1.3], gap="medium")

    with col_map:
        # PyDeck Interactive Map pinpointing Indian Agricultural Zones
        map_df = pd.DataFrame([
            {"name": k, "lat": v["lat"], "lon": v["lon"], "color": [46, 125, 50, 200] if k == selected_reg_name else [150, 150, 150, 140], "radius": 90000 if k == selected_reg_name else 50000}
            for k, v in AGRO_REGIONS.items()
        ])

        view_state = pdk.ViewState(
            latitude=22.5,
            longitude=78.5,
            zoom=3.8,
            pitch=35
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{name}"},
                map_style=None
            )
        )
        st.caption("🟢 Green pin highlights the currently inspected agricultural zone.")

    with col_details:
        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{selected_reg_name}</h3>
            <p><b>📍 Primary States:</b> {reg_data['states']}</p>
            <p><b>🌱 Supported Crops in Zone:</b> <b>{', '.join([c.replace('_', ' ') for c in reg_data['crops']])}</b></p>
            <p><b>🌍 Soil Type:</b> {reg_data['soil']}</p>
            <p><b>💧 Irrigation System:</b> {reg_data['irrigation']}</p>
            <p><b>☀️ Climate Dynamics:</b> {reg_data['climate']}</p>
            <div style="background: rgba(46, 125, 50, 0.08); padding: 12px; border-radius: 8px; border-left: 4px solid #2e7d32; margin-top: 10px;">
                <b>⚠️ High-Risk Disease Vulnerabilities:</b><br/>
                {reg_data['threats']}<br/><br/>
                <b>🔬 Biological Trigger:</b> {reg_data['why']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Live Leaf Preview Carousel for Crops in Selected Zone
    st.markdown(f"#### 🌿 Live Leaf Library for {selected_reg_name.split('(')[0]}")
    crop_cols = st.columns(len(reg_data["crops"]))
    for i, c_name in enumerate(reg_data["crops"]):
        with crop_cols[i]:
            healthy_img = get_healthy_reference_image(c_name)
            if healthy_img:
                st.image(healthy_img.resize((180, 180)), caption=f"{c_name.replace('_', ' ')} (Certified Leaf)", use_container_width=True)
            else:
                st.caption(f"{c_name.replace('_', ' ')}")

    st.markdown("---")

    # 4. Immediate CTA Banner to Diagnosis
    cta_col1, cta_col2 = st.columns([2.5, 1])
    with cta_col1:
        st.markdown("""
        ### Ready to analyze a leaf from your field?
        Switch directly to the **Leaf Diagnosis** engine to screen for 38 conditions with Grad-CAM heatmaps and export a PDF advisory.
        """)
    with cta_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.info("👉 Use the **Sidebar Menu** to open **🩺 Leaf Diagnosis**")


# ==========================================
# PAGE 1: DIAGNOSIS & TREATMENT (CORE APP)
# ==========================================
elif page == t["nav_diagnose"]:
    # 1. Intuitive 3-Step Workflow Banner (Clean, Practical, Symmetrical)
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{t['step1_title']}</div>
            <div class="step-text">{t['step1_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{t['step2_title']}</div>
            <div class="step-text">{t['step2_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{t['step3_title']}</div>
            <div class="step-text">{t['step3_desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. How to Test Expander (Clean, out of the way, but easily accessible)
    with st.expander(t["test_guide_title"], expanded=False):
        st.markdown(f"""
        {t['test_guide_1']}  
        {t['test_guide_2']}  
        {t['test_guide_3']}  
        """)

    # 3. Symmetrical, Centered Upload & Capture Area
    _, center_col, _ = st.columns([0.2, 0.6, 0.2])
    with center_col:
        mode = st.radio(
            "Input Mode",
            [t["upload_opt"], t["camera_opt"]],
            horizontal=True,
            label_visibility="collapsed"
        )
        if mode == t["upload_opt"]:
            file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
        else:
            file = st.camera_input(t["camera_label"])

    # 4. Diagnostic Results (Perfect 50/50 Symmetrical Layout)
    if file:
        img = Image.open(file).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        is_leaf, leaf_prob = detector.predict(x)

        # Gatekeeper Check
        if not is_leaf:
            st.error(t["not_a_leaf"].format(leaf_prob=leaf_prob))
            c_err, _ = st.columns([1, 2])
            c_err.image(img.resize((240, 240)), caption=t["your_photo"])
        else:
            heat, idx, conf, probs = gradcam(x)
            raw_class = CLASSES[idx]
            info = remedies[raw_class]
            crop_name = raw_class.split("___")[0]

            # Confidence Threshold
            if conf < THRESHOLD:
                st.warning(t["not_sure"].format(conf=conf))
                c_warn, _ = st.columns([1, 2])
                c_warn.image(img.resize((240, 240)), caption=t["your_photo"])
            else:
                st.markdown("---")
                
                # Prominent Result Callout
                st.markdown(f"""
                <div class="cs-result-banner">
                    <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; color:#2e7d32; font-weight:700;">Diagnostic Result</div>
                    <div class="cs-result-title">{info['name']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Equal 50/50 Symmetrical Split
                res_left, res_right = st.columns([1, 1], gap="large")

                with res_left:
                    # Key Metrics
                    sub_m1, sub_m2 = st.columns(2)
                    sub_m1.metric(t["conf_label"], f"{conf:.1%}")
                    sub_m2.metric(t["type_label"], info["type"].title())

                    if raw_class.endswith("___healthy"):
                        st.success(t["healthy_msg"])

                    # Symmetrical Side-by-Side Images
                    i1, i2 = st.columns(2)
                    i1.image(img.resize((224, 224)), caption=t["your_photo"], use_container_width=True)
                    heatmap_overlay_img = Image.fromarray((overlay(img, heat) * 255).astype(np.uint8))
                    i2.image(heatmap_overlay_img, caption=t["heatmap"], use_container_width=True)

                    # Top-3 Predictions Bar
                    st.markdown(f"**{t['top3']}**")
                    top3_idx = np.argsort(probs)[::-1][:3]
                    top3_list = []
                    for rank, i in enumerate(top3_idx, start=1):
                        p = float(probs[i])
                        cls_name = remedies[CLASSES[i]]["name"]
                        top3_list.append((cls_name, p))
                        st.progress(p, text=f"{rank}. {cls_name} — {p:.1%}")

                    st.markdown("<br/>", unsafe_allow_html=True)
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

                with res_right:
                    # Clean Tabs for Actionable Advice
                    tab_rx, tab_diag, tab_prev = st.tabs([
                        f"💊 {t['treatment']}",
                        f"🔬 {t['symptoms']}",
                        f"🛡️ {t['prevention']}"
                    ])

                    with tab_rx:
                        st.markdown(f"#### {t['treatment']}")
                        for s in info["treatment"]:
                            st.markdown(f"- **{s}**")

                    with tab_diag:
                        st.markdown(f"#### {t['cause']}")
                        st.write(info["cause"])
                        st.markdown(f"#### {t['symptoms']}")
                        st.write(info["symptoms"])

                    with tab_prev:
                        st.markdown(f"#### {t['prevention']}")
                        for s in info["prevention"]:
                            st.markdown(f"- **{s}**")

                    # Healthy Reference Comparison Card
                    healthy_ref_img = get_healthy_reference_image(crop_name)
                    if healthy_ref_img and not raw_class.endswith("___healthy"):
                        with st.expander(t["ref_heading"], expanded=False):
                            c_r1, c_r2 = st.columns(2)
                            c_r1.image(healthy_ref_img.resize((200, 200)), caption=f"Certified Healthy {crop_name.replace('_', ' ')}")
                            c_r2.image(img.resize((200, 200)), caption=f"Submitted Leaf ({info['name']})")

                st.caption(t["disclaimer"])


# ==========================================
# PAGE 2: SUPPORTED CROPS DIRECTORY
# ==========================================
elif page == t["nav_crops"]:
    st.subheader("🌱 Supported Crops & Pathogen Library (38 Classes)")
    st.write("Browse all 14 crops and 38 conditions recognized by CropShield:")

    crops = sorted(list(set(c.split("___")[0] for c in CLASSES)))
    chosen_crop = st.selectbox("Select Crop to Explore:", crops)

    crop_classes = [c for c in CLASSES if c.startswith(chosen_crop + "___")]
    
    cols = st.columns(len(crop_classes) if len(crop_classes) <= 3 else 3)
    for idx, c in enumerate(crop_classes):
        with cols[idx % 3]:
            info = remedies[c]
            st.markdown(f"""
            <div class="cs-card">
                <h4 style="margin-top:0; color:#2e7d32;">{info['name']}</h4>
                <b>Type:</b> {info['type'].title()}<br/><br/>
                <p style="font-size:0.9rem; opacity:0.9;">{info['symptoms']}</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: BENCHMARK METRICS & ARCHITECTURE (RESUME READY)
# ==========================================
elif page == t["nav_about"]:
    st.subheader("📊 Rigorous Empirical Benchmarks & Architecture")
    st.write("Complete experimental validation across studio lab benchmark (PlantVillage) and real-world in-situ field validation (PlantDoc):")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔬 Studio Benchmark (PlantVillage — 54,305 Images)")
        st.dataframe({
            "Model": ["SimpleCNN (Baseline)", "ResNet18 (Pretrained)"],
            "Parameters": ["399,142", "11,196,006"],
            "Test Accuracy": ["95.47%", "99.69%"],
            "Test Mistakes": ["246", "17 / 5,431"],
            "Macro F1": ["0.942", "0.996"]
        }, hide_index=True)

    with c2:
        st.markdown("#### 🌾 Real-World Field Adaptation (PlantDoc — 2,575 Images)")
        st.dataframe({
            "Metric": ["Top-1 Accuracy", "Top-3 Accuracy", "Crop-Level Identification", "PlantVillage Top-1"],
            "Before Fine-Tuning": ["20.17%", "43.28%", "46.64%", "99.69%"],
            "After Field Adaptation": ["61.34% (+41.2 pp)", "87.39% (+44.1 pp)", "79.41% (+32.8 pp)", "98.47% (Retained)"]
        }, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🛡️ LeafDetector: Out-of-Distribution Rejection Architecture")
    st.write("""
    - **Backbone:** Frozen ImageNet ResNet18 feature extractor (512-d embeddings).
    - **Head:** L2-regularized binary logistic regression.
    - **Performance:** **99.05% binary accuracy**, **0.9998 ROC-AUC**, rejecting 98.01% of non-leaf photos (faces, vehicles, objects) while maintaining 99.16% recall on real field leaves.
    """)


# ==========================================
# PAGE 4: KVK DIRECTORY & HELPLINES
# ==========================================
elif page == t["nav_kvk"]:
    st.subheader("🏛️ Krishi Vigyan Kendra (KVK) Directory & Farmer Helplines")
    st.write("Direct agricultural assistance centers and toll-free advisory numbers by state:")

    state_choice = st.selectbox(t["select_state"], list(kvk_directory.keys()))
    kvk_selected = kvk_directory[state_choice]

    col_kvk_card, _ = st.columns([2, 1])
    with col_kvk_card:
        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{kvk_selected['kvk']}</h3>
            <p><b>📍 Districts Covered:</b> {kvk_selected['district']}</p>
            <p><b>📞 Contact:</b> {kvk_selected['contact']}</p>
            <p style="background: rgba(46, 125, 50, 0.12); padding: 12px; border-radius: 8px;">
                <b>🌾 Kisan Call Center:</b> <span style="font-size:1.1rem; font-weight:700; color:#b71c1c;">{kvk_selected['helpline']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
