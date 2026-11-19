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
        "short_name": "🍎 Himalayan Belt",
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
        "short_name": "🌽 Indo-Gangetic Plains",
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
        "short_name": "🍇 Deccan Plateau",
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
        "short_name": "🍊 Coastal Horticulture",
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

# Detailed Crop Catalog for all 14 species supported in CropShield
CROP_CATALOG = {
    "Tomato": {
        "hindi": "टमाटर",
        "states": "Andhra Pradesh, Madhya Pradesh, Karnataka, Gujarat, Odisha",
        "season": "Year-round (Kharif, Rabi, & Zaid)",
        "soil": "Well-drained fertile loamy soil (pH 6.0–7.0)",
        "diseases": [
            "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Spider Mites", "Target Spot",
            "Yellow Leaf Curl Virus", "Mosaic Virus"
        ],
        "economic_fact": "India produces ~21 million MT annually; essential culinary staple across all states."
    },
    "Potato": {
        "hindi": "आलू",
        "states": "Uttar Pradesh, West Bengal, Bihar, Punjab, Gujarat",
        "season": "Rabi (sown Oct–Nov, harvested Jan–March)",
        "soil": "Loose, friable sandy loam rich in organic matter (pH 5.2–6.4)",
        "diseases": ["Early Blight (Alternaria)", "Late Blight (Phytophthora)"],
        "economic_fact": "India is the 2nd largest potato producer in the world (~54 million MT)."
    },
    "Corn_(maize)": {
        "hindi": "मक्का (Corn)",
        "states": "Karnataka, Madhya Pradesh, Maharashtra, Bihar, Rajasthan",
        "season": "Kharif (monsoon rainfed) & Rabi (irrigated)",
        "soil": "Deep fertile alluvial loam to red loamy soil",
        "diseases": ["Cercospora Leaf Spot (Gray Leaf Spot)", "Common Rust", "Northern Leaf Blight"],
        "economic_fact": "3rd most vital cereal crop in India, yielding ~34 million MT for food and poultry feed."
    },
    "Grape": {
        "hindi": "अंगूर",
        "states": "Maharashtra (Nashik, Sangli, Solapur), Karnataka, Andhra Pradesh",
        "season": "Pruning in Oct, harvest from Feb to April",
        "soil": "Well-drained basaltic black soil and sandy loam (pH 6.5–8.0)",
        "diseases": ["Black Rot", "Esca (Black Measles)", "Leaf Blight (Isariopsis)"],
        "economic_fact": "Maharashtra produces >80% of India's table grapes, earning major export revenue."
    },
    "Apple": {
        "hindi": "सेब",
        "states": "Jammu & Kashmir, Himachal Pradesh, Uttarakhand",
        "season": "Temperate winter chill, August–October harvest",
        "soil": "Well-drained loamy, slightly acidic soil (pH 5.5–6.5)",
        "diseases": ["Apple Scab (Venturia)", "Black Rot", "Cedar Apple Rust"],
        "economic_fact": "J&K and HP produce ~98% of India's commercial apple harvest."
    },
    "Pepper,_bell": {
        "hindi": "शिमला मिर्च (Capsicum)",
        "states": "Karnataka, Himachal Pradesh, Maharashtra, Haryana",
        "season": "Polyhouses year-round; outdoor Kharif/Rabi",
        "soil": "Rich sandy loam with good drainage (pH 6.0–6.8)",
        "diseases": ["Bacterial Spot"],
        "economic_fact": "Major commercial cash crop in modern micro-irrigation polyhouses."
    },
    "Orange": {
        "hindi": "संतरा (Citrus)",
        "states": "Maharashtra (Nagpur), Punjab (Kinnow), MP, Andhra Pradesh",
        "season": "Ambia Bahar (Feb–Mar) & Mrig Bahar (Jun–Jul)",
        "soil": "Light loamy to sub-clay soil, neutral pH with deep root zone",
        "diseases": ["Citrus Greening (Huanglongbing)"],
        "economic_fact": "Nagpur Mandarin and Punjab Kinnow are premier citrus hubs in India."
    },
    "Soybean": {
        "hindi": "सोयाबीन",
        "states": "Madhya Pradesh ('Soy State'), Maharashtra, Rajasthan",
        "season": "Kharif (June–July sowing with monsoon onset)",
        "soil": "Medium to deep black cotton soil (Vertisols)",
        "diseases": ["Healthy leaf baseline screening"],
        "economic_fact": "Critical protein and domestic edible oil crop covering over 12 million hectares."
    },
    "Strawberry": {
        "hindi": "स्ट्रॉबेरी",
        "states": "Maharashtra (Mahabaleshwar), Himachal Pradesh, Meghalaya",
        "season": "October planting, Dec–March fruiting",
        "soil": "Sandy loam enriched with organic compost under plastic mulch",
        "diseases": ["Leaf Scorch"],
        "economic_fact": "Mahabaleshwar plateau alone accounts for ~85% of India's commercial strawberries."
    },
    "Peach": {
        "hindi": "आड़ू (Peach)",
        "states": "Uttarakhand, Himachal Pradesh, Jammu & Kashmir, Punjab hills",
        "season": "Spring blossoming, May–July harvest",
        "soil": "Deep sandy loam with excellent internal drainage",
        "diseases": ["Bacterial Spot"],
        "economic_fact": "Key stone fruit crop in sub-Himalayan valleys and temperate zones."
    },
    "Cherry": {
        "hindi": "चेरी",
        "states": "Jammu & Kashmir (Srinagar, Shopian), Himachal Pradesh",
        "season": "Early spring flowering, May–June harvest",
        "soil": "Deep, rich, moisture-retentive fertile loam",
        "diseases": ["Powdery Mildew"],
        "economic_fact": "Kashmir Valley cultivates over 95% of India's commercial sweet cherries."
    },
    "Squash": {
        "hindi": "कद्दू / तोरी (Squash)",
        "states": "Uttar Pradesh, Punjab, Karnataka, Bihar",
        "season": "Zaid (Summer) & Kharif",
        "soil": "Warm, sandy loam rich in organic matter",
        "diseases": ["Powdery Mildew"],
        "economic_fact": "Rapid 60-day turnover cucurbit, popular for cash flow among vegetable growers."
    },
    "Blueberry": {
        "hindi": "ब्लूबेरी",
        "states": "Tamil Nadu (Nilgiris), Karnataka, Himachal Pradesh",
        "season": "Year-round controlled cultivation",
        "soil": "Highly acidic peat and pine bark mix (pH 4.5–5.2)",
        "diseases": ["Healthy leaf baseline screening"],
        "economic_fact": "High-margin superfood being pioneered in South Indian hill stations."
    },
    "Raspberry": {
        "hindi": "रसभरी (Raspberry)",
        "states": "Himachal Pradesh, Nilgiris, Kashmir Valley",
        "season": "Summer to early autumn",
        "soil": "Compost-rich, well-aerated sandy loam",
        "diseases": ["Healthy leaf baseline screening"],
        "economic_fact": "Niche premium berry for domestic bakery and gourmet markets."
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

@st.cache_data
def get_india_svg(selected_region: str) -> str:
    map_file = Path("app/india_map.json")
    if not map_file.exists():
        return "<p>Map data unavailable.</p>"
    with open(map_file, encoding="utf-8") as f:
        data = json.load(f)

    himalayan = {"Himachal Pradesh", "Jammu and Kashmir", "Uttarakhand"}
    indo_gangetic = {
        "Punjab", "Haryana", "Chandigarh", "Delhi", "Uttar Pradesh",
        "Bihar", "Madhya Pradesh", "West Bengal", "Jharkhand", "Chhattisgarh", "Rajasthan"
    }
    deccan = {
        "Maharashtra", "Gujarat", "Goa", "Dadra and Nagar Haveli",
        "Daman and Diu", "Karnataka", "Telangana"
    }

    ZONE_COLORS = {
        "🍎 Himalayan Temperate Zone (North)": "#0288d1",
        "🌽 Indo-Gangetic Plains (North & Central)": "#e65100",
        "🍇 Deccan Plateau & Western Semiarid Belt": "#7b1fa2",
        "🍊 Subtropical & Coastal Horticulture Belts": "#2e7d32",
    }
    active_color = ZONE_COLORS.get(selected_region, "#2e7d32")

    path_items = []
    for loc in data["locations"]:
        name = loc["name"]
        d = loc["path"]
        loc_id = loc["id"]

        if name in himalayan:
            z = "🍎 Himalayan Temperate Zone (North)"
        elif name in indo_gangetic:
            z = "🌽 Indo-Gangetic Plains (North & Central)"
        elif name in deccan:
            z = "🍇 Deccan Plateau & Western Semiarid Belt"
        else:
            z = "🍊 Subtropical & Coastal Horticulture Belts"

        is_active = (z == selected_region)

        if is_active:
            fill = active_color
            fill_op = "0.95"
            stroke = "#ffffff"
            sw = "1.8"
        else:
            fill = "#90a4ae"
            fill_op = "0.35"
            stroke = "#607d8b"
            sw = "0.8"

        title_text = f"{name} &#10;Farming Belt: {z.split('(')[0].strip()}"
        path_items.append(
            f'<path d="{d}" id="{loc_id}" fill="{fill}" fill-opacity="{fill_op}" '
            f'stroke="{stroke}" stroke-width="{sw}" style="transition: all 0.3s ease; cursor: pointer;">'
            f'<title>{title_text}</title>'
            f'</path>'
        )

    paths_str = "\n".join(path_items)

    svg_markup = f"""
    <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 10px; padding: 14px 10px 10px 10px; text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 612 696" width="100%" height="auto" style="max-height: 440px; display: block; margin: auto;">
            <g id="india-map-group">
                {paths_str}
            </g>
        </svg>
        <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.95;">
            <b>Active Farming Belt:</b> <span style="color:{active_color}; font-weight:700;">{selected_region.split('(')[0].strip()}</span>
        </div>
        <div style="font-size: 0.78rem; opacity: 0.7; margin-top: 2px;">
            Hover any state to inspect name & agricultural zone
        </div>
    </div>
    """
    return svg_markup

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
# PAGE 0: INTRO - INDIAN FARMING EXPLORER
# ==========================================
if page == t["nav_overview"]:
    st.subheader("🗺️ Agro-Climatic Regions & Crop Farming Systems of India")
    st.write(
        "India is characterized by diverse agricultural micro-climates. Explore how the **14 key crops supported by CropShield** "
        "are cultivated, seasonal threats evolve, and simulate disease outbreaks across India's premier farming belts."
    )

    # 1. Real-Time Agricultural Pulse Metrics
    st.markdown("#### 📡 National Agricultural Situation & System Overview")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="🌾 Active Cropping Cycle",
            value="Rabi / Zaid",
            delta="Summer Sowing Active",
            delta_color="normal"
        )
    with col_m2:
        st.metric(
            label="🌱 Monitored Crop Species",
            value="14 Crops",
            delta="Staples & Horticulture",
            delta_color="off"
        )
    with col_m3:
        st.metric(
            label="🛡️ Diagnostic Pathologies",
            value="38 States",
            delta="Bacterial, Fungal, Viral",
            delta_color="off"
        )
    with col_m4:
        st.metric(
            label="🎯 Field Robustness (Top-3)",
            value="87.4%",
            delta="+44.1 pp over studio",
            delta_color="normal"
        )

    st.markdown("---")

    # 2. Cropping Seasons Matrix (Kharif, Rabi, Zaid)
    st.markdown("#### 🌦️ Cropping Seasons of India (Kharif, Rabi, Zaid)")
    st.caption("Select a cropping season to inspect how temperature, monsoon rainfall, and pathogen risks shift throughout the year:")

    tab_kharif, tab_rabi, tab_zaid = st.tabs([
        "🌧️ Kharif (Monsoon: June – October)",
        "❄️ Rabi (Winter: October – March)",
        "☀️ Zaid (Summer: March – June)"
    ])

    with tab_kharif:
        st.markdown("""
        <div class="cs-card">
            <h4 style="color:#2e7d32; margin-top:0;">🌧️ Kharif Season (South-West Monsoon Crops)</h4>
            <p><b>Sowing:</b> June–July (with monsoon arrival) | <b>Harvest:</b> September–October</p>
            <p><b>Key Supported Crops:</b> <b>Corn (Maize), Soybean, Tomato, Bell Pepper, Squash</b></p>
            <p><b>Weather Dynamics:</b> High ambient temperatures (28°C–36°C) coupled with continuous rainfall and relative humidity >85%.</p>
            <div style="background: rgba(46, 125, 50, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #2e7d32; margin-top: 8px;">
                <b>⚠️ Dominant Pathogen Risks:</b><br/>
                Continuous leaf moisture fosters rapid spore germination of <i>Corn Northern Leaf Blight</i>, <i>Tomato Septoria & Bacterial Spot</i>, and <i>Soybean foliar blights</i>. Splash dispersal via raindrops is the chief vector.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_rabi:
        st.markdown("""
        <div class="cs-card">
            <h4 style="color:#1565c0; margin-top:0;">❄️ Rabi Season (Winter & Post-Monsoon Crops)</h4>
            <p><b>Sowing:</b> October–November | <b>Harvest:</b> February–April</p>
            <p><b>Key Supported Crops:</b> <b>Potato, Apple (dormancy/flowering), Grape (pruning), Peach, Cherry</b></p>
            <p><b>Weather Dynamics:</b> Cool, crisp weather (5°C–20°C), dense early-morning fog, and heavy ground dew deposition.</p>
            <div style="background: rgba(21, 101, 192, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #1565c0; margin-top: 8px;">
                <b>⚠️ Dominant Pathogen Risks:</b><br/>
                Persistent leaf wetness (>10 hours) and cool fog create optimal incubation for <i>Potato Late Blight (Phytophthora infestans)</i> and <i>Apple Scab</i>. Unchecked, Late Blight can devastate an entire field in 5–7 days.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_zaid:
        st.markdown("""
        <div class="cs-card">
            <h4 style="color:#e65100; margin-top:0;">☀️ Zaid Season (Warm Irrigated Summer Crops)</h4>
            <p><b>Sowing:</b> March | <b>Harvest:</b> May–June</p>
            <p><b>Key Supported Crops:</b> <b>Squash, Tomato (irrigated), Bell Pepper (polyhouse), Strawberry (hill stations)</b></p>
            <p><b>Weather Dynamics:</b> Intense solar radiation, high temperatures (32°C–42°C), low relative humidity (<40%), high evaporation rates.</p>
            <div style="background: rgba(230, 81, 0, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #e65100; margin-top: 8px;">
                <b>⚠️ Dominant Pathogen Risks:</b><br/>
                Hot and dry conditions suppress Phytophthora but trigger explosive populations of <i>Two-Spotted Spider Mites</i> and sap-sucking whitefly vectors spreading <i>Tomato Yellow Leaf Curl Virus (TYLCV)</i>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Regional Agro-Climatic Belts & Focused Map of India
    st.markdown("#### 🗺️ Regional Agro-Climatic Belts of India")
    
    # Session state for belt selection (click map or button)
    if "selected_region" not in st.session_state or st.session_state["selected_region"] not in AGRO_REGIONS:
        st.session_state["selected_region"] = list(AGRO_REGIONS.keys())[1]

    # Quick Selection Buttons for Farming Belts
    st.markdown("📍 **Select Agricultural Belt to Highlight on the Map of India:**")
    b_cols = st.columns(4)
    for i, (r_key, r_val) in enumerate(AGRO_REGIONS.items()):
        with b_cols[i]:
            is_active = (r_key == st.session_state["selected_region"])
            if st.button(
                r_val.get("short_name", r_key.split("(")[0].strip()),
                key=f"belt_btn_{i}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state["selected_region"] = r_key
                st.rerun()

    selected_reg_name = st.session_state["selected_region"]
    reg_data = AGRO_REGIONS[selected_reg_name]

    col_map, col_details = st.columns([1.1, 1.3], gap="medium")

    with col_map:
        st.markdown(get_india_svg(selected_reg_name), unsafe_allow_html=True)

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

    # Live Leaf Preview for Crops in Zone
    st.markdown(f"##### 🌿 Certified Leaf Baselines for {selected_reg_name.split('(')[0]}")
    crop_cols = st.columns(len(reg_data["crops"]))
    for i, c_name in enumerate(reg_data["crops"]):
        with crop_cols[i]:
            healthy_img = get_healthy_reference_image(c_name)
            if healthy_img:
                st.image(healthy_img.resize((180, 180)), caption=f"{c_name.replace('_', ' ')} (Certified Leaf)", use_container_width=True)
            else:
                st.caption(f"{c_name.replace('_', ' ')}")

    st.markdown("---")

    # 4. Micro-Climate Disease Outbreak Simulator (Real-Time Pathology Predictor)
    st.markdown("#### 🔬 Real-Time Micro-Climate Outbreak Simulator")
    st.write(
        "Plant pathogens (fungal spores, bacteria, and viral vectors) require specific micro-climatic thresholds to germinate and spread. "
        "Adjust the environmental sliders below to simulate live field conditions and calculate the disease outbreak probability:"
    )

    sim_c1, sim_c2 = st.columns([1.2, 1.2], gap="large")

    with sim_c1:
        st.markdown("##### ⚙️ Environmental Controls")
        sim_crop = st.selectbox(
            "Select Crop to Simulate:",
            ["Potato", "Tomato", "Apple", "Grape", "Corn_(maize)", "Pepper,_bell", "Orange", "Squash"],
            index=0
        )
        sim_temp = st.slider("🌡️ Daytime Temperature (°C):", min_value=10, max_value=42, value=18, step=1)
        sim_humidity = st.slider("💧 Relative Humidity (%):", min_value=20, max_value=100, value=85, step=1)
        sim_wet_hours = st.slider("🌿 Continuous Leaf Wetness / Dew (Hours):", min_value=0, max_value=24, value=10, step=1)

    with sim_c2:
        st.markdown("##### 📊 Pathological Outbreak Calculation")
        # Agronomic pathology simulation based on Wallin / Smith criteria
        risk_score = 15
        pathogen_name = "None detected (Normal micro-flora)"
        alert_level = "LOW"
        action_advice = "Field conditions are within safe agronomic parameters. Continue standard routine scouting."

        if sim_crop in ["Potato", "Tomato"]:
            if 12 <= sim_temp <= 22 and sim_humidity >= 80 and sim_wet_hours >= 8:
                risk_score = min(98, 40 + int(sim_wet_hours * 3.5) + int((sim_humidity - 70) * 1.5))
                pathogen_name = "Late Blight (Phytophthora infestans)"
                alert_level = "CRITICAL"
                action_advice = "🚨 High outbreak risk! Late blight sporangia germinate rapidly in cool, dew-saturated foliage. Apply protective copper or systemic fungicide before full canopy infection."
            elif 22 <= sim_temp <= 30 and sim_humidity >= 70 and sim_wet_hours >= 4:
                risk_score = min(88, 30 + int(sim_wet_hours * 3.0) + int((sim_humidity - 60) * 1.2))
                pathogen_name = "Early Blight (Alternaria solani)"
                alert_level = "MODERATE"
                action_advice = "⚠️ Alternaria conidia require alternating wet and warm periods. Remove lower infected leaves and avoid overhead irrigation."
            elif sim_temp >= 32 and sim_humidity <= 50:
                risk_score = 72
                pathogen_name = "Spider Mites / Whitefly TYLCV Vector"
                alert_level = "MODERATE"
                action_advice = "Hot, dry conditions accelerate mite reproduction and whitefly migration. Monitor underside of leaves."

        elif sim_crop in ["Apple", "Cherry"]:
            if 14 <= sim_temp <= 24 and sim_humidity >= 75 and sim_wet_hours >= 6:
                risk_score = min(92, 35 + int(sim_wet_hours * 3.5))
                pathogen_name = "Apple Scab (Venturia inaequalis)"
                alert_level = "CRITICAL"
                action_advice = "🚨 Mills Period met: Ascospores discharge during prolonged leaf wetness. Fungicidal protection needed within 48 hours."
            elif 18 <= sim_temp <= 28 and 50 <= sim_humidity <= 75 and sim_wet_hours < 4:
                risk_score = 65
                pathogen_name = "Powdery Mildew (Podosphaera leucotricha)"
                alert_level = "MODERATE"
                action_advice = "Mildew conidia germinate in warm, humid air without free leaf moisture. Prune dense inner canopy for airflow."

        elif sim_crop == "Grape":
            if 20 <= sim_temp <= 30 and sim_humidity >= 75 and sim_wet_hours >= 6:
                risk_score = min(90, 40 + int(sim_wet_hours * 3.2))
                pathogen_name = "Black Rot (Guignardia bidwellii)"
                alert_level = "CRITICAL"
                action_advice = "🚨 Warm rain and leaf wetness trigger pycnidiospore infection on young leaves and berry clusters."
            else:
                risk_score = 30

        elif sim_crop == "Corn_(maize)":
            if 18 <= sim_temp <= 28 and sim_humidity >= 80 and sim_wet_hours >= 6:
                risk_score = 78
                pathogen_name = "Northern Corn Leaf Blight (Exserohilum turcicum)"
                alert_level = "MODERATE"
                action_advice = "Heavy dews and moderate temperatures favor cigar-shaped necrotic lesions. Ensure crop rotation and residue management."
            else:
                risk_score = 25

        elif sim_crop == "Pepper,_bell":
            if 24 <= sim_temp <= 32 and sim_humidity >= 80 and sim_wet_hours >= 5:
                risk_score = 84
                pathogen_name = "Bacterial Spot (Xanthomonas campestris)"
                alert_level = "CRITICAL"
                action_advice = "Bacterial pathogens enter via stomata during wind-driven rain and high heat. Avoid handling plants when foliage is wet."
            else:
                risk_score = 30
        else:
            if sim_humidity >= 80 and sim_wet_hours >= 6:
                risk_score = 60
                pathogen_name = "General Foliar Fungal Complex"
                alert_level = "MODERATE"
                action_advice = "High moisture creates potential for fungal germination. Monitor leaf margins."

        # Display Result Gauge
        if alert_level == "CRITICAL":
            st.error(f"🚨 **Calculated Outbreak Risk: {risk_score}% — CRITICAL ALERT**")
        elif alert_level == "MODERATE":
            st.warning(f"⚠️ **Calculated Outbreak Risk: {risk_score}% — MODERATE RISK**")
        else:
            st.success(f"🟢 **Calculated Outbreak Risk: {risk_score}% — LOW / SAFE**")

        st.progress(risk_score / 100.0)
        st.markdown(f"**🔬 Predicted Pathogen:** `{pathogen_name}`")
        st.info(f"**💡 Agronomic Advisory:** {action_advice}")

    st.markdown("---")

    # 5. 14-Crop Deep-Dive Directory
    st.markdown("#### 📚 Crop Pathology & Agronomy Library (14 Species)")
    st.write("Browse all 14 crop species supported by CropShield to view their state distribution, soil preferences, and disease profiles:")

    inspect_crop = st.selectbox(
        "Choose Crop to Inspect:",
        list(CROP_CATALOG.keys()),
        format_func=lambda x: f"{x.replace('_', ' ')} ({CROP_CATALOG[x]['hindi']})"
    )
    c_info = CROP_CATALOG[inspect_crop]

    c_c1, c_c2 = st.columns([1, 2], gap="medium")
    with c_c1:
        c_img = get_healthy_reference_image(inspect_crop)
        if c_img:
            st.image(c_img.resize((240, 240)), caption=f"Certified Healthy {inspect_crop.replace('_', ' ')} Leaf", use_container_width=True)
        else:
            st.info("Certified Leaf Baseline")
    with c_c2:
        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{inspect_crop.replace('_', ' ')} ({c_info['hindi']})</h3>
            <p><b>📍 Premier Growing States:</b> {c_info['states']}</p>
            <p><b>🗓️ Primary Season:</b> {c_info['season']}</p>
            <p><b>🌍 Preferred Soil & pH:</b> {c_info['soil']}</p>
            <p><b>📈 Economic Fact:</b> {c_info['economic_fact']}</p>
            <p><b>🛡️ Monitored Pathologies in CropShield:</b></p>
            <ul>
                {''.join([f'<li><b>{d}</b></li>' for d in c_info['diseases']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 6. Immediate CTA Banner to Diagnosis
    cta_col1, cta_col2 = st.columns([2.5, 1])
    with cta_col1:
        st.markdown("""
        ### Ready to analyze a leaf from your field?
        Switch directly to the **Leaf Diagnosis** engine to screen for 38 conditions with Grad-CAM heatmaps and export an official PDF advisory.
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
