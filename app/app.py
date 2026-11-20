import io
from datetime import datetime
from pathlib import Path
import json
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import streamlit as st
import torch

from dataset import CLASSES, eval_tf
from model import build_resnet18
from gradcam import GradCAM
from leaf_detector import load_leaf_detector
from report_pdf import generate_pdf_report

st.set_page_config(
    page_title="CropShield — Crop Disease & Pest Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    .cs-hero {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.02) 100%);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 26px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        position: relative;
        overflow: hidden;
    }
    
    .cs-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.25);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .cs-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 9999px;
        background-color: #10b981;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    }
    
    .cs-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #059669;
        margin: 0 0 6px 0;
        letter-spacing: -0.8px;
    }
    
    .cs-tagline {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.85;
        margin: 0;
        font-weight: 500;
        line-height: 1.5;
    }
    
    .step-box {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 18px;
        padding: 20px 18px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
    }
    
    .step-box:hover {
        border-color: rgba(16, 185, 129, 0.35);
        transform: translateY(-2px);
    }
    
    .step-num {
        font-size: 1.1rem;
        font-weight: 800;
        color: #059669;
        margin-bottom: 6px;
    }
    
    .step-text {
        font-size: 0.88rem;
        color: var(--text-color);
        opacity: 0.85;
        line-height: 1.45;
    }
    
    .cs-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 8px 24px -8px rgba(0, 0, 0, 0.03);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .cs-card:hover {
        border-color: rgba(16, 185, 129, 0.35);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 12px 28px -6px rgba(16, 185, 129, 0.05);
    }
    
    .cs-result-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(5, 150, 105, 0.04) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 18px;
        padding: 18px 24px;
        margin-bottom: 22px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .cs-result-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #059669;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.16) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(16, 185, 129, 0.3) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #059669 !important;
        letter-spacing: -0.5px !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        opacity: 0.85 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
    
    button[data-baseweb="tab"] {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
    }
    
    img {
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.14) !important;
    }
</style>
""", unsafe_allow_html=True)

UI = {
    "English": {
        "title": "🌿 CropShield AI",
        "tagline": "Empowering Indian Agriculture with Computer Vision & Targeted Crop Health Diagnostics",
        "nav_overview": "🌾 Indian Agriculture & Farming Belts",
        "nav_diagnose": "🩺 Leaf Diagnosis",
        "nav_crops": "🔬 Crop Pathology & Simulator",
        "nav_kvk": "🏛️ KVK & Farmer Helplines",
        "sidebar_nav": "🌿 CropShield Navigation",
        "sidebar_help_title": "🌾 National Farmer Helpline",
        "sidebar_help_desc": "📞 **1800-180-1551**\n\nToll-Free Kisan Call Center • Available in 22 Regional Languages (6 AM – 10 PM)",
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
        "diag_result_title": "Diagnostic Result",
        "healthy_cert_caption": "Certified Healthy {crop}",
        "submitted_leaf_caption": "Submitted Leaf ({name})",
        "p0_subhead": "🗺️ Agro-Climatic Regions & Crop Farming Systems of India",
        "p0_desc": "India is characterized by diverse agricultural micro-climates. Explore how the **14 key crops supported by CropShield** are cultivated, seasonal threats evolve, and simulate disease outbreaks across India's premier farming belts.",
        "p0_metrics_head": "📡 National Agricultural Situation & System Overview",
        "p0_m1_label": "🌾 Active Cropping Cycle",
        "p0_m1_val": "Rabi / Zaid",
        "p0_m1_delta": "Summer Sowing Active",
        "p0_m2_label": "🌱 Monitored Crop Species",
        "p0_m2_val": "14 Crops",
        "p0_m2_delta": "Staples & Horticulture",
        "p0_m3_label": "🛡️ Diagnostic Pathologies",
        "p0_m3_val": "38 Classes",
        "p0_m3_delta": "Bacterial, Fungal, Viral",
        "p0_m4_label": "🎯 Field Robustness (Top-3)",
        "p0_m4_val": "87.4%",
        "p0_m4_delta": "+44.1 pp over studio",
        "p0_seasons_head": "🌦️ Cropping Seasons of India (Kharif, Rabi, Zaid)",
        "p0_seasons_caption": "Select a cropping season to inspect how temperature, monsoon rainfall, and pathogen risks shift throughout the year:",
        "p0_tab_kharif": "🌧️ Kharif (Monsoon: June – October)",
        "p0_tab_rabi": "❄️ Rabi (Winter: October – March)",
        "p0_tab_zaid": "☀️ Zaid (Summer: March – June)",
        "p0_kharif_title": "🌧️ Kharif Season (South-West Monsoon Crops)",
        "p0_kharif_p1": "<b>Sowing:</b> June–July (with monsoon arrival) | <b>Harvest:</b> September–October",
        "p0_kharif_p2": "<b>Key Supported Crops:</b> <b>Corn (Maize), Soybean, Tomato, Bell Pepper, Squash</b>",
        "p0_kharif_p3": "<b>Weather Dynamics:</b> High ambient temperatures (28°C–36°C) coupled with continuous rainfall and relative humidity >85%.",
        "p0_kharif_warn": "<b>⚠️ Dominant Pathogen Risks:</b><br/>Continuous leaf moisture fosters rapid spore germination of <i>Corn Northern Leaf Blight</i>, <i>Tomato Septoria & Bacterial Spot</i>, and <i>Soybean foliar blights</i>. Splash dispersal via raindrops is the chief vector.",
        "p0_rabi_title": "❄️ Rabi Season (Winter & Post-Monsoon Crops)",
        "p0_rabi_p1": "<b>Sowing:</b> October–November | <b>Harvest:</b> February–April",
        "p0_rabi_p2": "<b>Key Supported Crops:</b> <b>Potato, Apple (dormancy/flowering), Grape (pruning), Peach, Cherry</b>",
        "p0_rabi_p3": "<b>Weather Dynamics:</b> Cool, crisp weather (5°C–20°C), dense early-morning fog, and heavy ground dew deposition.",
        "p0_rabi_warn": "<b>⚠️ Dominant Pathogen Risks:</b><br/>Persistent leaf wetness (>10 hours) and cool fog create optimal incubation for <i>Potato Late Blight (Phytophthora infestans)</i> and <i>Apple Scab</i>. Unchecked, Late Blight can devastate an entire field in 5–7 days.",
        "p0_zaid_title": "☀️ Zaid Season (Warm Irrigated Summer Crops)",
        "p0_zaid_p1": "<b>Sowing:</b> March | <b>Harvest:</b> May–June",
        "p0_zaid_p2": "<b>Key Supported Crops:</b> <b>Squash, Tomato (irrigated), Bell Pepper (polyhouse), Strawberry (hill stations)</b>",
        "p0_zaid_p3": "<b>Weather Dynamics:</b> Intense solar radiation, high temperatures (32°C–42°C), low relative humidity (<40%), high evaporation rates.",
        "p0_zaid_warn": "<b>⚠️ Dominant Pathogen Risks:</b><br/>Hot and dry conditions suppress Phytophthora but trigger explosive populations of <i>Two-Spotted Spider Mites</i> and sap-sucking whitefly vectors spreading <i>Tomato Yellow Leaf Curl Virus (TYLCV)</i>.",
        "p0_belts_head": "🗺️ Regional Agro-Climatic Belts of India",
        "p0_belts_select": "📍 **Select Agricultural Belt to Highlight on the Map of India:**",
        "p0_card_states": "📍 Primary States:",
        "p0_card_crops": "🌱 Supported Crops in Zone:",
        "p0_card_soil": "🌍 Soil Type:",
        "p0_card_irrigation": "💧 Irrigation System:",
        "p0_card_climate": "☀️ Climate Dynamics:",
        "p0_card_threats": "⚠️ High-Risk Disease Vulnerabilities:",
        "p0_card_why": "🔬 Biological Trigger:",
        "p0_cta1_title": "🩺 Field Leaf Diagnosis Engine",
        "p0_cta1_desc": "Screen crop leaves with <b>LeafDetector OOD rejection</b> and <b>Grad-CAM lesion heatmaps</b>. Get verified guidance and download an official advisory PDF.",
        "p0_cta1_link": '👉 Select "🩺 Leaf Diagnosis" from the sidebar menu',
        "p0_cta2_title": "🔬 Crop Pathology & Simulator",
        "p0_cta2_desc": "Access the real-time micro-climate risk calculator (Wallin/Smith criteria), certified leaf baselines, and complete 38-disease library.",
        "p0_cta2_link": '👉 Select "🔬 Crop Pathology & Simulator" from the sidebar menu',
        "p2_subhead": "🔬 Crop Pathology, Outbreak Simulator & 38-Disease Library",
        "p2_desc": "Explore real-time micro-climatic disease outbreak dynamics, certified healthy leaf references, and complete pathological profiles across all 14 crop species and 38 conditions recognized by CropShield:",
        "p2_sim_head": "🌡️ Real-Time Micro-Climate Outbreak Simulator",
        "p2_sim_desc": "Foliar pathogens (fungal spores, bacteria, and viral vectors) require specific micro-climatic thresholds to germinate and infect. Adjust the environmental sliders below to simulate field conditions and calculate the outbreak probability:",
        "p2_ctrl_head": "⚙️ Environmental Field Controls",
        "p2_crop_sim_label": "Select Crop to Simulate:",
        "p2_temp_label": "🌡️ Daytime Temperature (°C):",
        "p2_hum_label": "💧 Relative Humidity (%):",
        "p2_wet_label": "🌿 Continuous Leaf Wetness / Dew (Hours):",
        "p2_calc_head": "📊 Pathological Outbreak Calculation",
        "p2_pred_pathogen": "🔬 Predicted Pathogen:",
        "p2_agri_advice": "💡 Agronomic Advisory:",
        "p2_risk_critical": "🚨 **Calculated Outbreak Risk: {risk}% — CRITICAL ALERT**",
        "p2_risk_mod": "⚠️ **Calculated Outbreak Risk: {risk}% — MODERATE RISK**",
        "p2_risk_low": "🟢 **Calculated Outbreak Risk: {risk}% — LOW / SAFE**",
        "p2_profile_head": "🌿 Crop Profile & Certified Leaf Baseline",
        "p2_profile_select": "Choose Crop to Inspect:",
        "p2_profile_caption": "Certified Healthy {crop} Leaf (Baseline)",
        "p2_profile_states": "📍 Premier Growing States:",
        "p2_profile_season": "🗓️ Primary Season:",
        "p2_profile_soil": "🌍 Preferred Soil & pH:",
        "p2_profile_econ": "📈 Economic Fact:",
        "p2_conditions_head": "🛡️ Pathological Conditions Screened for {crop}",
        "p2_conditions_caption": "Standardized uniform visual samples, biological classification, foliar symptoms, and cultural actions for each condition:",
        "p2_card_type": "Pathogen Type:",
        "p2_card_symptoms": "Visible Symptoms:",
        "p2_card_prev": "Cultural Prevention:",
        "p3_subhead": "🏛️ Krishi Vigyan Kendra (KVK) Directory & Farmer Helplines",
        "p3_desc": "Direct agricultural assistance centers and toll-free advisory numbers by state:",
        "p3_districts": "📍 Districts Covered:",
        "p3_contact": "📞 Contact:",
        "p3_kcc": "🌾 Kisan Call Center:",
    },
    "हिन्दी (Hindi)": {
        "title": "🌿 क्रॉपशील्ड (CropShield AI)",
        "tagline": "भारतीय कृषि के लिए कंप्यूटर विज़न आधारित फसल रोग पहचान एवं उपचार प्रणाली",
        "nav_overview": "🌾 भारतीय कृषि एवं कृषि क्षेत्र",
        "nav_diagnose": "🩺 पत्ती जाँच व निदान",
        "nav_crops": "🔬 फसल रोग विज्ञान (Pathology)",
        "nav_kvk": "🏛️ कृषि विज्ञान केंद्र (KVK)",
        "sidebar_nav": "🌿 क्रॉपशील्ड नेविगेशन",
        "sidebar_help_title": "🌾 राष्ट्रीय किसान हेल्पलाइन",
        "sidebar_help_desc": "📞 **1800-180-1551**\n\nटोल-फ्री किसान कॉल सेंटर • 22 क्षेत्रीय भाषाओं में उपलब्ध (सुबह 6 बजे – रात 10 बजे)",
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
        "diag_result_title": "जाँच परिणाम",
        "healthy_cert_caption": "प्रमाणित स्वस्थ {crop}",
        "submitted_leaf_caption": "जाँच हेतु प्रस्तुत पत्ती ({name})",
        "p0_subhead": "🗺️ भारत के कृषि-जलवायु क्षेत्र एवं फसल प्रणालियाँ",
        "p0_desc": "भारत विविध कृषि-जलवायु परिस्थितियों से संपन्न है। जानें कि क्रॉपशील्ड समर्थित 14 मुख्य फसलें कैसे उगाई जाती हैं, मौसमी खतरे कैसे बदलते हैं, और प्रमुख कृषि क्षेत्रों में बीमारियों का प्रकोप कैसे फैलता है:",
        "p0_metrics_head": "📡 राष्ट्रीय कृषि स्थिति एवं सिस्टम का संक्षिप्त विवरण",
        "p0_m1_label": "🌾 वर्तमान फसल चक्र",
        "p0_m1_val": "रबी / जायद",
        "p0_m1_delta": "ग्रीष्मकालीन बुवाई सक्रिय",
        "p0_m2_label": "🌱 समर्थित फसल प्रजातियाँ",
        "p0_m2_val": "14 फसलें",
        "p0_m2_delta": "अनाज व बागवानी",
        "p0_m3_label": "🛡️ रोग श्रेणियां",
        "p0_m3_val": "38 श्रेणियां (Classes)",
        "p0_m3_delta": "जीवाणु, फफूंद, विषाणु",
        "p0_m4_label": "🎯 खेत में मॉडल सटीकता (Top-3)",
        "p0_m4_val": "87.4%",
        "p0_m4_delta": "प्रयोगशाला से +44.1 pp अधिक",
        "p0_seasons_head": "🌦️ भारत के मुख्य फसल मौसम (खरीफ, रबी, जायद)",
        "p0_seasons_caption": "तापमान, मानसूनी वर्षा और रोग के जोखिम को देखने के लिए फसल का मौसम चुनें:",
        "p0_tab_kharif": "🌧️ खरीफ (मानसून: जून – अक्टूबर)",
        "p0_tab_rabi": "❄️ रबी (सर्दियां: अक्टूबर – मार्च)",
        "p0_tab_zaid": "☀️ जायद (गर्मी: मार्च – जून)",
        "p0_kharif_title": "🌧️ खरीफ मौसम (दक्षिण-पश्चिम मानसूनी फसलें)",
        "p0_kharif_p1": "<b>बुवाई:</b> जून–जुलाई (मानसून आगमन पर) | <b>कटाई:</b> सितंबर–अक्टूबर",
        "p0_kharif_p2": "<b>मुख्य समर्थित फसलें:</b> <b>मक्का, सोयाबीन, टमाटर, शिमला मिर्च, कद्दू</b>",
        "p0_kharif_p3": "<b>मौसम की स्थिति:</b> उच्च तापमान (28°C–36°C), निरंतर वर्षा और सापेक्ष आर्द्रता >85%।",
        "p0_kharif_warn": "<b>⚠️ मुख्य रोग जोखिम:</b><br/>पत्तियों पर लगातार नमी से <i>मक्के का पत्ता झुलसा</i>, <i>टमाटर का जीवाणु धब्बा व सेप्टोरिया</i>, और <i>सोयाबीन का झुलसा रोग</i> तेजी से पनपते हैं। बारिश की बूंदों के छींटे इन रोगों को तेजी से फैलाते हैं।",
        "p0_rabi_title": "❄️ रबी मौसम (सर्दियां एवं मानसून पश्चात फसलें)",
        "p0_rabi_p1": "<b>बुवाई:</b> अक्टूबर–नवंबर | <b>कटाई:</b> फरवरी–अप्रैल",
        "p0_rabi_p2": "<b>मुख्य समर्थित फसलें:</b> <b>आलू, सेब, अंगूर (छंटाई), आड़ू, चेरी</b>",
        "p0_rabi_p3": "<b>मौसम की स्थिति:</b> ठंडी व शुष्क जलवायु (5°C–20°C), सुबह का घना कोहरा और भारी ओस का जमाव।",
        "p0_rabi_warn": "<b>⚠️ मुख्य रोग जोखिम:</b><br/>पत्तियों पर 10+ घंटे ओस और ठंड <i>आलू के पछेती झुलसा (Late Blight)</i> और <i>सेब के स्कैब</i> के लिए आदर्श स्थितियां बनाती हैं। ध्यान न देने पर पछेती झुलसा 5–7 दिनों में पूरा खेत नष्ट कर सकता है।",
        "p0_zaid_title": "☀️ जायद मौसम (ग्रीष्मकालीन सिंचित फसलें)",
        "p0_zaid_p1": "<b>बुवाई:</b> मार्च | <b>कटाई:</b> मई–जून",
        "p0_zaid_p2": "<b>मुख्य समर्थित फसलें:</b> <b>कद्दू/तोरी, सिंचित टमाटर, पॉलीहाउस शिमला मिर्च, स्ट्रॉबेरी</b>",
        "p0_zaid_p3": "<b>मौसम की स्थिति:</b> तेज धूप, अत्यधिक तापमान (32°C–42°C), कम नमी (<40%), तेज वाष्पीकरण।",
        "p0_zaid_warn": "<b>⚠️ मुख्य रोग जोखिम:</b><br/>शुष्क गर्मी में <i>लाल मकड़ी (Spider Mites)</i> और रस चूसने वाली सफेद मक्खी के हमले बढ़ते हैं, जो <i>टमाटर का पीली पत्ती मरोड़िया रोग (TYLCV)</i> फैलाती हैं।",
        "p0_belts_head": "🗺️ भारत के प्रमुख कृषि-जलवायु क्षेत्र",
        "p0_belts_select": "📍 **भारत के मानचित्र पर देखने के लिए कृषि क्षेत्र चुनें:**",
        "p0_card_states": "📍 प्रमुख राज्य:",
        "p0_card_crops": "🌱 क्षेत्र की मुख्य फसलें:",
        "p0_card_soil": "🌍 मिट्टी का प्रकार:",
        "p0_card_irrigation": "💧 सिंचाई व्यवस्था:",
        "p0_card_climate": "☀️ जलवायु स्थिति:",
        "p0_card_threats": "⚠️ उच्च जोखिम वाली बीमारियां:",
        "p0_card_why": "🔬 जैविक कारण:",
        "p0_cta1_title": "🩺 खेत में पत्ती जाँच व निदान प्रणाली",
        "p0_cta1_desc": "<b>LeafDetector</b> द्वारा गैर-पत्ती की पहचान और <b>Grad-CAM हीटमैप</b> से रोग की सटीक पुष्टि। सत्यापित सलाह देखें व आधिकारिक पीडीएफ रिपोर्ट डाउनलोड करें।",
        "p0_cta1_link": '👉 साइडबार मेनू से "🩺 पत्ती जाँच व निदान" चुनें',
        "p0_cta2_title": "🔬 फसल रोग विज्ञान एवं प्रकोप सिमुलेटर",
        "p0_cta2_desc": "वास्तविक समय सूक्ष्म-जलवायु जोखिम कैलकुलेटर, प्रमाणित स्वस्थ पत्तियों के संदर्भ, और संपूर्ण 38-रोग संदर्भ पुस्तकालय का उपयोग करें।",
        "p0_cta2_link": '👉 साइडबार मेनू से "🔬 फसल रोग विज्ञान (Pathology)" चुनें',
        "p2_subhead": "🔬 फसल रोग विज्ञान, प्रकोप सिमुलेटर एवं 38-रोग संदर्भ पुस्तकालय",
        "p2_desc": "सूक्ष्म-जलवायु रोग प्रकोप की स्थिति, प्रमाणित स्वस्थ पत्तियों के संदर्भ, और क्रॉपशील्ड समर्थित सभी 14 फसलों एवं 38 स्थितियों का संपूर्ण रोग विज्ञान देखें:",
        "p2_sim_head": "🌡️ वास्तविक समय सूक्ष्म-जलवायु प्रकोप सिमुलेटर (Outbreak Simulator)",
        "p2_sim_desc": "पत्तियों के रोगज़नक़ों (फफूंद, जीवाणु, विषाणु) के अंकुरण के लिए विशिष्ट मौसम की आवश्यकता होती है। खेत की परिस्थितियों का अनुकरण करने और रोग के प्रकोप की संभावना जानने के लिए नीचे दिए गए स्लाइडर्स को बदलें:",
        "p2_ctrl_head": "⚙️ खेत का वातावरण एवं मौसम नियंत्रण",
        "p2_crop_sim_label": "सिमुलेशन के लिए फसल चुनें:",
        "p2_temp_label": "🌡️ दिन का तापमान (°C):",
        "p2_hum_label": "💧 सापेक्ष आर्द्रता / नमी (%):",
        "p2_wet_label": "🌿 पत्तियों पर लगातार गीलापन / ओस (घंटे):",
        "p2_calc_head": "📊 रोग प्रकोप की गणना एवं विश्लेषण",
        "p2_pred_pathogen": "🔬 संभावित रोगज़नक़:",
        "p2_agri_advice": "💡 कृषि सलाह / क्या करें:",
        "p2_risk_critical": "🚨 **प्रकोप का अनुमानित जोखिम: {risk}% — अत्यधिक गंभीर चेतावनी (CRITICAL)**",
        "p2_risk_mod": "⚠️ **प्रकोप का अनुमानित जोखिम: {risk}% — मध्यम जोखिम (MODERATE)**",
        "p2_risk_low": "🟢 **प्रकोप का अनुमानित जोखिम: {risk}% — कम जोखिम / सुरक्षित (SAFE)**",
        "p2_profile_head": "🌿 फसल का विवरण एवं प्रमाणित स्वस्थ पत्ती का संदर्भ",
        "p2_profile_select": "जाँच के लिए फसल चुनें:",
        "p2_profile_caption": "प्रमाणित स्वस्थ {crop} पत्ती (आधार संदर्भ)",
        "p2_profile_states": "📍 प्रमुख उत्पादक राज्य:",
        "p2_profile_season": "🗓️ मुख्य मौसम / बुवाई:",
        "p2_profile_soil": "🌍 उपयुक्त मिट्टी एवं pH:",
        "p2_profile_econ": "📈 आर्थिक व कृषि महत्व:",
        "p2_conditions_head": "🛡️ {crop} के लिए जांची जाने वाली रोग स्थितियाँ",
        "p2_conditions_caption": "प्रत्येक स्थिति के लिए समान मानक दृश्य नमूने, जैविक वर्गीकरण, लक्षण और रोकथाम के उपाय:",
        "p2_card_type": "रोगज़नक़ का प्रकार:",
        "p2_card_symptoms": "दिखाई देने वाले लक्षण:",
        "p2_card_prev": "रोकथाम के उपाय:",
        "p3_subhead": "🏛️ कृषि विज्ञान केंद्र (KVK) निर्देशिका एवं किसान हेल्पलाइन",
        "p3_desc": "राज्यवार प्रत्यक्ष कृषि सहायता केंद्र एवं टोल-फ्री किसान परामर्श नंबर:",
        "p3_districts": "📍 संबंधित जिले:",
        "p3_contact": "📞 संपर्क / पता:",
        "p3_kcc": "🌾 किसान कॉल सेंटर:",
    }
}

AGRO_REGIONS = {
    "🍎 Himalayan Temperate Zone (North)": {
        "short_name": "🍎 Himalayan Belt",
        "short_name_hi": "🍎 हिमालयी क्षेत्र",
        "name_hi": "🍎 हिमालयी शीतोष्ण क्षेत्र (उत्तर)",
        "states": "Himachal Pradesh, Jammu & Kashmir, Uttarakhand",
        "states_hi": "हिमाचल प्रदेश, जम्मू और कश्मीर, उत्तराखंड",
        "lat": 32.0, "lon": 76.5,
        "crops": ["Apple", "Peach", "Cherry"],
        "crops_hi": ["सेब", "आड़ू", "चेरी"],
        "soil": "Brown hill soil, acidic mountain soil",
        "soil_hi": "भूरा पहाड़ी व अम्लीय पर्वतीय मिट्टी (pH 5.5–6.5)",
        "irrigation": "Natural mountain springs, rainfed contours",
        "irrigation_hi": "प्राकृतिक पर्वतीय झरने, सीढ़ीदार वर्षा आधारित प्रणाली",
        "climate": "Chilling winters, temperate summers (15°C–28°C)",
        "climate_hi": "कड़ाके की ठंड, मध्यम गर्मियां (15°C–28°C), शीतोष्ण जलवायु",
        "threats": "Apple Scab (Venturia), Cedar Apple Rust, Powdery Mildew",
        "threats_hi": "सेब का स्कैब (Venturia), देवदार रतुआ (Cedar Apple Rust), चूर्णिल आसिता (Powdery Mildew)",
        "why": "Spring snow-melt creates high ambient humidity, triggering fungal spore release.",
        "why_hi": "वसंत में बर्फ पिघलने से हवा में उच्च नमी बनती है, जिससे फफूंद के बीजाणु तेजी से फैलते हैं।"
    },
    "🌽 Indo-Gangetic Plains (North & Central)": {
        "short_name": "🌽 Indo-Gangetic Plains",
        "short_name_hi": "🌽 गंगा का मैदान",
        "name_hi": "🌽 सिंधु-गंगा का मैदानी क्षेत्र (उत्तर व मध्य)",
        "states": "Uttar Pradesh, Punjab, Haryana, Bihar, Madhya Pradesh",
        "states_hi": "उत्तर प्रदेश, पंजाब, हरियाणा, बिहार, मध्य प्रदेश",
        "lat": 26.8, "lon": 80.9,
        "crops": ["Potato", "Corn_(maize)", "Soybean", "Tomato"],
        "crops_hi": ["आलू", "मक्का", "सोयाबीन", "टमाटर"],
        "soil": "Alluvial deep fertile loam (Khadar and Bhangar)",
        "soil_hi": "जलोढ़ उपजाऊ गहरी दोमट मिट्टी (खादर और बांगर)",
        "irrigation": "Extensive canal networks, tube-wells",
        "irrigation_hi": "नहरों का व्यापक जाल, नलकूप (ट्यूबवेल)",
        "climate": "Extreme summers (40°C+), cold humid winters (5°C–18°C)",
        "climate_hi": "अत्यधिक गर्मियां (40°C+), ठंडी व नम सर्दियां (5°C–18°C)",
        "threats": "Late Blight (Phytophthora) in potato/tomato, Corn Leaf Blight, Early Blight",
        "threats_hi": "आलू व टमाटर का पछेती झुलसा (Late Blight), मक्के का पत्ता झुलसा, अगेती झुलसा",
        "why": "Heavy winter morning dew and dense fog create perfect incubation for Phytophthora water mould.",
        "why_hi": "सर्दियों की सुबह भारी ओस और घना कोहरा फाइटोफ्थोरा फफूंद के संक्रमण के लिए आदर्श वातावरण बनाते हैं।"
    },
    "🍇 Deccan Plateau & Western Semiarid Belt": {
        "short_name": "🍇 Deccan Plateau",
        "short_name_hi": "🍇 दक्कन का पठार",
        "name_hi": "🍇 दक्कन का पठार एवं पश्चिमी अर्ध-शुष्क क्षेत्र",
        "states": "Maharashtra, Karnataka, Telangana, Gujarat",
        "states_hi": "महाराष्ट्र, कर्नाटक, तेलंगाना, गुजरात",
        "lat": 19.5, "lon": 74.5,
        "crops": ["Grape", "Pepper,_bell", "Tomato", "Squash"],
        "crops_hi": ["अंगूर", "शिमला मिर्च", "टमाटर", "कद्दू/तोरी"],
        "soil": "Black cotton soil (Regur), volcanic basaltic loam",
        "soil_hi": "काली कपासी मिट्टी (रेगुर), बेसाल्टिक उपजाऊ दोमट मिट्टी",
        "irrigation": "Precision drip systems, farm ponds (Tals)",
        "irrigation_hi": "ड्रिप सिंचाई प्रणाली, खेत के तालाब (शेत्तले)",
        "climate": "Semi-arid, tropical wet-and-dry (20°C–36°C)",
        "climate_hi": "अर्ध-शुष्क, उष्णकटिबंधीय नम व शुष्क जलवायु (20°C–36°C)",
        "threats": "Grape Black Rot, Esca (Black Measles), Bell Pepper Bacterial Spot",
        "threats_hi": "अंगूर का काला सड़न (Black Rot), एस्का रोग, शिमला मिर्च का जीवाणु धब्बा",
        "why": "Sudden unseasonal post-monsoon showers cause rapid berry cracking and fungal entry.",
        "why_hi": "मानसून के बाद अचानक होने वाली बेमौसम बारिश से अंगूर चटक जाते हैं और फंगस तेजी से प्रवेश करता है।"
    },
    "🍊 Subtropical & Coastal Horticulture Belts": {
        "short_name": "🍊 Coastal Horticulture",
        "short_name_hi": "🍊 तटीय बागवानी",
        "name_hi": "🍊 उप-उष्णकटिबंधीय एवं तटीय बागवानी क्षेत्र",
        "states": "Punjab/Rajasthan (Citrus), Andhra Pradesh, Tamil Nadu, Mahabaleshwar",
        "states_hi": "पंजाब/राजस्थान (किन्नू/संतरा), आंध्र प्रदेश, तमिलनाडु, महाबलेश्वर",
        "lat": 15.5, "lon": 78.5,
        "crops": ["Orange", "Strawberry", "Blueberry", "Raspberry"],
        "crops_hi": ["संतरा", "स्ट्रॉबेरी", "ब्लूबेरी", "रसभरी"],
        "soil": "Red sandy loam, laterite, enriched peat for berries",
        "soil_hi": "लाल रेतीली दोमट, लेटराइट, बेरी के लिए समृद्ध पीट मिट्टी",
        "irrigation": "Micro-sprinklers, raised beds with silver-black mulch",
        "irrigation_hi": "माइक्रो-स्प्रिंकलर, प्लास्टिक मल्च वाले ऊंचे बेड",
        "climate": "Subtropical, frost-free winters, high sunshine hours",
        "climate_hi": "उप-उष्णकटिबंधीय, पाला-रहित सर्दियां, भरपूर धूप",
        "threats": "Citrus Greening (Huanglongbing), Strawberry Leaf Scorch, Spider Mites",
        "threats_hi": "सिट्रस ग्रीनिंग (Huanglongbing), स्ट्रॉबेरी पत्ती झुलसा, लाल मकड़ी (Spider Mites)",
        "why": "Sap-sucking psyllid insects and warm dry air accelerate vector-borne viral/bacterial spread.",
        "why_hi": "रस चूसने वाले कीट (साइलिड/सफेद मक्खी) और गर्म शुष्क हवा से विषाणु व जीवाणु तेजी से फैलते हैं।"
    }
}

CROP_CATALOG = {
    "Tomato": {
        "hindi": "टमाटर",
        "states": "Andhra Pradesh, Madhya Pradesh, Karnataka, Gujarat, Odisha",
        "states_hi": "आंध्र प्रदेश, मध्य प्रदेश, कर्नाटक, गुजरात, ओडिशा",
        "season": "Year-round (Kharif, Rabi, & Zaid)",
        "season_hi": "वर्ष भर (खरीफ, रबी और जायद तीनों मौसम)",
        "soil": "Well-drained fertile loamy soil (pH 6.0–7.0)",
        "soil_hi": "अच्छी जल निकासी वाली उपजाऊ दोमट मिट्टी (pH 6.0–7.0)",
        "diseases": [
            "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Spider Mites", "Target Spot",
            "Yellow Leaf Curl Virus", "Mosaic Virus"
        ],
        "economic_fact": "India produces ~21 million MT annually; essential culinary staple across all states.",
        "economic_fact_hi": "भारत में प्रतिवर्ष लगभग 2.1 करोड़ मीट्रिक टन उत्पादन; सभी राज्यों की रसोई का अनिवार्य हिस्सा।"
    },
    "Potato": {
        "hindi": "आलू",
        "states": "Uttar Pradesh, West Bengal, Bihar, Punjab, Gujarat",
        "states_hi": "उत्तर प्रदेश, पश्चिम बंगाल, बिहार, पंजाब, गुजरात",
        "season": "Rabi (sown Oct–Nov, harvested Jan–March)",
        "season_hi": "रबी (अक्टूबर-नवंबर में बुवाई, जनवरी-मार्च में खुदाई)",
        "soil": "Loose, friable sandy loam rich in organic matter (pH 5.2–6.4)",
        "soil_hi": "जीवांश युक्त हल्की बलुई दोमट मिट्टी (pH 5.2–6.4)",
        "diseases": ["Early Blight (Alternaria)", "Late Blight (Phytophthora)"],
        "economic_fact": "India is the 2nd largest potato producer in the world (~54 million MT).",
        "economic_fact_hi": "भारत दुनिया का दूसरा सबसे बड़ा आलू उत्पादक देश है (लगभग 5.4 करोड़ मीट्रिक टन)।"
    },
    "Corn_(maize)": {
        "hindi": "मक्का (Corn)",
        "states": "Karnataka, Madhya Pradesh, Maharashtra, Bihar, Rajasthan",
        "states_hi": "कर्नाटक, मध्य प्रदेश, महाराष्ट्र, बिहार, राजस्थान",
        "season": "Kharif (monsoon rainfed) & Rabi (irrigated)",
        "season_hi": "खरीफ (वर्षा आधारित) और रबी (सिंचित)",
        "soil": "Deep fertile alluvial loam to red loamy soil",
        "soil_hi": "गहरी उपजाऊ जलोढ़ दोमट से लाल दोमट मिट्टी",
        "diseases": ["Cercospora Leaf Spot (Gray Leaf Spot)", "Common Rust", "Northern Leaf Blight"],
        "economic_fact": "3rd most vital cereal crop in India, yielding ~34 million MT for food and poultry feed.",
        "economic_fact_hi": "भारत की तीसरी सबसे महत्वपूर्ण खाद्यान्न फसल, भोजन और पोल्ट्री फीड के लिए ~3.4 करोड़ टन उत्पादन।"
    },
    "Grape": {
        "hindi": "अंगूर",
        "states": "Maharashtra (Nashik, Sangli, Solapur), Karnataka, Andhra Pradesh",
        "states_hi": "महाराष्ट्र (नासिक, सांगली, सोलापुर), कर्नाटक, आंध्र प्रदेश",
        "season": "Pruning in Oct, harvest from Feb to April",
        "season_hi": "अक्टूबर में छंटाई (प्रूनिंग), फरवरी से अप्रैल में तुड़ाई",
        "soil": "Well-drained basaltic black soil and sandy loam (pH 6.5–8.0)",
        "soil_hi": "अच्छी जल निकास वाली काली व रेतीली दोमट मिट्टी (pH 6.5–8.0)",
        "diseases": ["Black Rot", "Esca (Black Measles)", "Leaf Blight (Isariopsis)"],
        "economic_fact": "Maharashtra produces >80% of India's table grapes, earning major export revenue.",
        "economic_fact_hi": "महाराष्ट्र भारत के 80% से अधिक अंगूर का उत्पादन करता है, जिससे भारी निर्यात राजस्व मिलता है।"
    },
    "Apple": {
        "hindi": "सेब",
        "states": "Jammu & Kashmir, Himachal Pradesh, Uttarakhand",
        "states_hi": "जम्मू और कश्मीर, हिमाचल प्रदेश, उत्तराखंड",
        "season": "Temperate winter chill, August–October harvest",
        "season_hi": "शीतोष्ण ठंडी जलवायु, अगस्त से अक्टूबर में तुड़ाई",
        "soil": "Well-drained loamy, slightly acidic soil (pH 5.5–6.5)",
        "soil_hi": "अच्छी जल निकासी वाली दोमट, हल्की अम्लीय मिट्टी (pH 5.5–6.5)",
        "diseases": ["Apple Scab (Venturia)", "Black Rot", "Cedar Apple Rust"],
        "economic_fact": "J&K and HP produce ~98% of India's commercial apple harvest.",
        "economic_fact_hi": "जम्मू-कश्मीर और हिमाचल प्रदेश मिलकर भारत के लगभग 98% सेब का उत्पादन करते हैं।"
    },
    "Pepper,_bell": {
        "hindi": "शिमला मिर्च (Capsicum)",
        "states": "Karnataka, Himachal Pradesh, Maharashtra, Haryana",
        "states_hi": "कर्नाटक, हिमाचल प्रदेश, महाराष्ट्र, हरियाणा",
        "season": "Polyhouses year-round; outdoor Kharif/Rabi",
        "season_hi": "पॉलीहाउस में वर्ष भर; खुले खेत में खरीफ व रबी",
        "soil": "Rich sandy loam with good drainage (pH 6.0–6.8)",
        "soil_hi": "उत्तम जल निकासी वाली उपजाऊ रेतीली दोमट (pH 6.0–6.8)",
        "diseases": ["Bacterial Spot"],
        "economic_fact": "Major commercial cash crop in modern micro-irrigation polyhouses.",
        "economic_fact_hi": "आधुनिक ड्रिप व पॉलीहाउस खेती में सबसे अधिक लाभ देने वाली नकदी फसल।"
    },
    "Orange": {
        "hindi": "संतरा (Citrus)",
        "states": "Maharashtra (Nagpur), Punjab (Kinnow), MP, Andhra Pradesh",
        "states_hi": "महाराष्ट्र (नागपुर), पंजाब (किन्नू), मध्य प्रदेश, आंध्र प्रदेश",
        "season": "Ambia Bahar (Feb–Mar) & Mrig Bahar (Jun–Jul)",
        "season_hi": "अम्बिया बहार (फरवरी-मार्च) और मृग बहार (जून-जुलाई)",
        "soil": "Light loamy to sub-clay soil, neutral pH with deep root zone",
        "soil_hi": "हल्की दोमट से मध्यम चिकनी, गहरे जड़ क्षेत्र वाली मिट्टी",
        "diseases": ["Healthy Leaf", "Citrus Greening (Huanglongbing)"],
        "economic_fact": "Nagpur Mandarin and Punjab Kinnow are premier citrus hubs in India.",
        "economic_fact_hi": "नागपुर का संतरा और पंजाब का किन्नू भारत के सबसे प्रसिद्ध साइट्रस केंद्र हैं।"
    },
    "Soybean": {
        "hindi": "सोयाबीन",
        "states": "Madhya Pradesh ('Soy State'), Maharashtra, Rajasthan",
        "states_hi": "मध्य प्रदेश ('सोया राज्य'), महाराष्ट्र, राजस्थान",
        "season": "Kharif (June–July sowing with monsoon onset)",
        "season_hi": "खरीफ (मानसून शुरू होने पर जून-जुलाई में बुवाई)",
        "soil": "Medium to deep black cotton soil (Vertisols)",
        "soil_hi": "मध्यम से गहरी काली कपासी मिट्टी (वर्टिसोल)",
        "diseases": ["Healthy Leaf", "Brown Spot (Septoria glycines)", "Asian Soybean Rust"],
        "economic_fact": "Critical protein and domestic edible oil crop covering over 12 million hectares.",
        "economic_fact_hi": "1.2 करोड़ हेक्टेयर में उगाई जाने वाली देश की प्रमुख प्रोटीन व खाद्य तेल फसल।"
    },
    "Strawberry": {
        "hindi": "स्ट्रॉबेरी",
        "states": "Maharashtra (Mahabaleshwar), Himachal Pradesh, Meghalaya",
        "states_hi": "महाराष्ट्र (महाबलेश्वर), हिमाचल प्रदेश, मेघालय",
        "season": "October planting, Dec–March fruiting",
        "season_hi": "अक्टूबर में रोपाई, दिसंबर से मार्च में फल",
        "soil": "Sandy loam enriched with organic compost under plastic mulch",
        "soil_hi": "कम्पोस्ट युक्त बलुई दोमट मिट्टी, प्लास्टिक मल्चिंग के साथ",
        "diseases": ["Healthy Leaf", "Leaf Scorch"],
        "economic_fact": "Mahabaleshwar plateau alone accounts for ~85% of India's commercial strawberries.",
        "economic_fact_hi": "अकेले महाबलेश्वर का पठार भारत की लगभग 85% व्यावसायिक स्ट्रॉबेरी का उत्पादन करता है।"
    },
    "Peach": {
        "hindi": "आड़ू (Peach)",
        "states": "Uttarakhand, Himachal Pradesh, Jammu & Kashmir, Punjab hills",
        "states_hi": "उत्तराखंड, हिमाचल प्रदेश, जम्मू और कश्मीर, पंजाब की पहाड़ियां",
        "season": "Summer (May to July harvest)",
        "season_hi": "ग्रीष्म ऋतु (मई से जुलाई में तुड़ाई)",
        "soil": "Deep sandy loam with good drainage (pH 5.8–6.8)",
        "soil_hi": "गहरी उपजाऊ बलुई दोमट मिट्टी (pH 5.8–6.8)",
        "diseases": ["Bacterial Spot", "Healthy Leaf"],
        "economic_fact": "Significant horticultural income generator for northern hill farmers.",
        "economic_fact_hi": "उत्तर भारत के पहाड़ी किसानों के लिए प्रमुख आय का स्रोत।"
    },
    "Cherry": {
        "hindi": "चेरी",
        "states": "Kashmir Valley, Himachal Pradesh (Kotgarh)",
        "states_hi": "कश्मीर घाटी, हिमाचल प्रदेश (कोटगढ़)",
        "season": "May to late June",
        "season_hi": "मई से जून के अंत तक",
        "soil": "Deep fertile loamy soil with excellent internal drainage (pH 6.0–7.0)",
        "soil_hi": "गहरी, अच्छी जल निकासी वाली उपजाऊ दोमट मिट्टी (pH 6.0–7.0)",
        "diseases": ["Powdery Mildew", "Healthy Leaf"],
        "economic_fact": "Kashmir supplies over 95% of India's commercial fresh cherry production.",
        "economic_fact_hi": "कश्मीर भारत की 95% से अधिक ताजी वाणिज्यिक चेरी की आपूर्ति करता है।"
    },
    "Squash": {
        "hindi": "कद्दू / तोरी (Squash)",
        "states": "Uttar Pradesh, Punjab, Bihar, Karnataka, Haryana",
        "states_hi": "उत्तर प्रदेश, पंजाब, बिहार, कर्नाटक, हरियाणा",
        "season": "Zaid (Feb–March) & Kharif (June–July)",
        "season_hi": "जायद (फरवरी-मार्च) और खरीफ (जून-जुलाई)",
        "soil": "Sandy loam to loamy soil rich in organic matter (pH 6.0–7.5)",
        "soil_hi": "हल्की बलुई दोमट से दोमट, जीवांश युक्त (pH 6.0–7.5)",
        "diseases": ["Healthy Leaf", "Powdery Mildew"],
        "economic_fact": "Fast-growing high-yield vegetable critical to regional weekly agricultural markets (Mandis).",
        "economic_fact_hi": "कम समय में तैयार होने वाली, भारतीय बाजारों में अत्यधिक मांग वाली सब्जी।"
    },
    "Blueberry": {
        "hindi": "ब्लूबेरी",
        "states": "Tamil Nadu (Ooty/Nilgiris), Himachal Pradesh",
        "states_hi": "तमिलनाडु (ऊटी/नीलगिरि), हिमाचल प्रदेश",
        "season": "Year-round controlled cultivation",
        "season_hi": "नियंत्रित वातावरण में वर्ष भर उत्पादन",
        "soil": "Highly acidic peat and pine bark mix (pH 4.5–5.2)",
        "soil_hi": "अत्यधिक अम्लीय पीट और पाइन छाल का मिश्रण (pH 4.5–5.2)",
        "diseases": ["Healthy Leaf", "Leaf Rust (Thekopsora minima)"],
        "economic_fact": "High-margin superfood being pioneered in South Indian hill stations.",
        "economic_fact_hi": "उच्च मूल्य वाला सुपरफूड, जिसकी खेती दक्षिण भारत में तेजी से बढ़ रही है।"
    },
    "Raspberry": {
        "hindi": "रसभरी (Raspberry)",
        "states": "Himachal Pradesh, Nilgiris, Kashmir Valley",
        "states_hi": "हिमाचल प्रदेश, नीलगिरि, कश्मीर घाटी",
        "season": "Summer to early autumn",
        "season_hi": "गर्मी से प्रारंभिक शरद ऋतु",
        "soil": "Compost-rich, well-aerated sandy loam",
        "soil_hi": "कम्पोस्ट युक्त, हवादार बलुई दोमट मिट्टी",
        "diseases": ["Healthy Leaf", "Leaf Spot (Sphaerulina rubi)"],
        "economic_fact": "Niche premium berry for domestic bakery and gourmet markets.",
        "economic_fact_hi": "बेकरी और प्रीमियम बाज़ार के लिए उच्च लाभ वाली विदेशी बेरी।"
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

def get_standardized_image(image_or_path, size=(240, 240)):
    from PIL import ImageOps
    if image_or_path is None:
        return None
    try:
        if isinstance(image_or_path, (str, Path)):
            p = Path(image_or_path)
            if not p.exists():
                return None
            img = Image.open(p).convert("RGB")
        elif isinstance(image_or_path, Image.Image):
            img = image_or_path.convert("RGB")
        else:
            return None
        return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    except Exception:
        return None

def get_healthy_reference_image(crop_name: str):
    prefix = "Cherry_(including_sour)___healthy" if crop_name == "Cherry" else f"{crop_name}___healthy"
    for ext in [".jpg", ".png", ".jpeg"]:
        p = Path("demo_images") / f"{prefix}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    c_sub = crop_name.split("_")[0]
    for ext in [".jpg", ".png"]:
        p = Path("demo_images") / f"{c_sub}___healthy{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None

EXTENDED_PATHOLOGY = {
    "Soybean___Brown_Spot": {
        "name": "Soybean Brown Spot (Septoria glycines)",
        "name_hi": "सोयाबीन — भूरा धब्बा रोग (Septoria glycines)",
        "type": "fungus",
        "type_hi": "फफूंद (Fungus)",
        "image": "demo_images/Soybean___Brown_Spot.jpg",
        "symptoms": "Irregular dark brown to black spots appearing on lower leaves, causing premature yellowing and leaf drop in wet seasons.",
        "symptoms_hi": "निचली पत्तियों पर अनियमित गहरे भूरे से काले धब्बे, जो नम मौसम में पत्तियों को पीला करके समय से पहले गिरा देते हैं।",
        "prevention": ["Rotate crops with non-legume cereals every 2 years.", "Plow under crop residues deeply post-harvest to eliminate overwintering spores."],
        "prevention_hi": ["हर 2 साल में गैर-दलहनी फसलों के साथ फसल चक्र अपनाएं।", "फसल अवशेषों को गहरी जुताई करके मिट्टी में दबा दें।"]
    },
    "Blueberry___Rust": {
        "name": "Blueberry Leaf Rust (Thekopsora minima)",
        "name_hi": "ब्लूबेरी — पत्ती रतुआ (Thekopsora minima)",
        "type": "fungus",
        "type_hi": "फफूंद (Fungus)",
        "image": "demo_images/Blueberry___Rust.jpg",
        "symptoms": "Small yellow chlorotic flecks on upper leaf surfaces accompanied by reddish-orange pustules on the leaf undersides.",
        "symptoms_hi": "पत्ती की ऊपरी सतह पर छोटे पीले धब्बे और निचली सतह पर लाल-नारंगी रंग के उभार (पस्ट्यूल्स)।",
        "prevention": ["Prune old canes to facilitate sunlight penetration and canopy aeration.", "Use root-zone drip irrigation to prevent foliar wetting."],
        "prevention_hi": ["पुरानी टहनियों की छंटाई करें ताकि धूप और हवा का संचार बना रहे।", "पत्तियों को गीला होने से बचाने के लिए ड्रिप सिंचाई का उपयोग करें।"]
    },
    "Raspberry___Leaf_Spot": {
        "name": "Raspberry Leaf Spot (Sphaerulina rubi)",
        "name_hi": "रसभरी — पत्ती धब्बा रोग (Sphaerulina rubi)",
        "type": "fungus",
        "type_hi": "फफूंद (Fungus)",
        "image": "demo_images/Raspberry___Leaf_Spot.jpg",
        "symptoms": "Circular spots with ash-gray centers and distinct purple-brown borders leading to premature foliage shedding.",
        "symptoms_hi": "राख जैसे भूरे केंद्र और गहरे बैंगनी किनारों वाले गोलाकार धब्बे, जिससे पत्तियां समय से पहले झड़ने लगती हैं।",
        "prevention": ["Prune out and destroy spent canes immediately after summer harvest.", "Maintain wide trellis row spacing for quick canopy drying."],
        "prevention_hi": ["गर्मी की तुड़ाई के तुरंत बाद पुरानी व सूखी टहनियों को काटकर नष्ट करें।", "कतारों के बीच पर्याप्त दूरी रखें ताकि पत्तियां जल्दी सूख सकें।"]
    },
    "Orange___healthy": {
        "name": "Orange (Citrus) — Healthy Leaf",
        "name_hi": "संतरा — स्वस्थ पत्ती",
        "type": "healthy",
        "type_hi": "स्वस्थ (Healthy)",
        "image": "demo_images/Orange___healthy.jpg",
        "symptoms": "Vigorous, dark-green glossy foliage with intact winged petioles and zero blotching or vein-yellowing.",
        "symptoms_hi": "चमकदार गहरे हरे रंग की स्वस्थ पत्तियां, जिन पर कोई पीलापन, धब्बा या सिकुड़न नहीं है।",
        "prevention": ["Maintain balanced NPK fertilization and micronutrient sprays.", "Scout regularly for sap-sucking psyllid vectors."],
        "prevention_hi": ["संतुलित एनपीके (NPK) खाद और सूक्ष्म पोषक तत्वों का छिड़काव बनाए रखें।", "रस चूसने वाले कीटों (साइलिड) की नियमित निगरानी करें।"]
    },
    "Squash___healthy": {
        "name": "Squash / Cucurbit — Healthy Leaf",
        "name_hi": "कद्दू / तोरी — स्वस्थ पत्ती",
        "type": "healthy",
        "type_hi": "स्वस्थ (Healthy)",
        "image": "demo_images/Squash___healthy.jpg",
        "symptoms": "Broad, vigorous dark-green lobed canopy without powdery fungal dust, mosaic mottling, or water-soaked lesions.",
        "symptoms_hi": "चौड़ी, गहरे हरे रंग की स्वस्थ पत्तियां, जिन पर कोई सफेद पाउडर, मोज़ेक धब्बे या गीले घाव नहीं हैं।",
        "prevention": ["Ensure drip irrigation at base of vines.", "Scout lower leaf surfaces early morning for powdery mildew conidia."],
        "prevention_hi": ["बेलों की जड़ों पर ड्रिप विधि से पानी दें।", "सुबह के समय निचली पत्तियों की नियमित जांच करें।"]
    }
}

@st.cache_data
def get_india_svg(selected_region: str, is_hindi: bool = False) -> str:
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

        belt_label = AGRO_REGIONS[z]["name_hi"] if is_hindi else z.split('(')[0].strip()
        state_tag = "कृषि क्षेत्र" if is_hindi else "Farming Belt"
        title_text = f"{name} &#10;{state_tag}: {belt_label}"
        path_items.append(
            f'<path d="{d}" id="{loc_id}" fill="{fill}" fill-opacity="{fill_op}" '
            f'stroke="{stroke}" stroke-width="{sw}" style="transition: all 0.3s ease; cursor: pointer;">'
            f'<title>{title_text}</title>'
            f'</path>'
        )

    paths_str = "\n".join(path_items)
    active_display = AGRO_REGIONS[selected_region]["name_hi"] if is_hindi else selected_region.split('(')[0].strip()
    active_txt = "सक्रिय कृषि क्षेत्र:" if is_hindi else "Active Farming Belt:"
    caption_txt = "भारत का वेक्टर मानचित्र • राज्य-वार भौगोलिक संरचना" if is_hindi else "Vector Map of India • Focused State Geometry"

    svg_markup = f"""
    <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.16); border-radius: 18px; padding: 18px 14px 14px 14px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 612 696" width="100%" height="auto" style="max-height: 440px; display: block; margin: auto;">
            <g id="india-map-group">
                {paths_str}
            </g>
        </svg>
        <div style="margin-top: 10px; font-size: 0.9rem; opacity: 0.95;">
            <b>{active_txt}</b> <span style="color:{active_color}; font-weight:700;">{active_display}</span>
        </div>
        <div style="font-size: 0.78rem; opacity: 0.7; margin-top: 2px;">
            {caption_txt}
        </div>
    </div>
    """
    return svg_markup

detector, gradcam, all_remedies, kvk_directory = load_assets()

with st.sidebar:
    lang = st.radio("Language / भाषा", ["English", "हिन्दी (Hindi)"], horizontal=True)
    is_hindi = (lang == "हिन्दी (Hindi)")
    t = UI[lang]
    remedies = all_remedies[lang]
    
    st.markdown(f"### {t['sidebar_nav']}")
    
    st.markdown("---")
    page = st.radio(
        "Select Portal Section",
        [t["nav_overview"], t["nav_diagnose"], t["nav_crops"], t["nav_kvk"]],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"#### {t['sidebar_help_title']}")
    st.info(t["sidebar_help_desc"])

badge_text = "फसल स्वास्थ्य निदान • 14 मुख्य फसलें" if is_hindi else "Autonomous Foliar Diagnostics • 14 Crop Species"
st.markdown(f"""
<div class="cs-hero">
    <div class="cs-hero-badge"><span class="cs-badge-dot"></span> {badge_text}</div>
    <div class="cs-title">{t['title']}</div>
    <div class="cs-tagline">{t['tagline']}</div>
</div>
""", unsafe_allow_html=True)


if page == t["nav_overview"]:
    st.subheader(t["p0_subhead"])
    st.write(t["p0_desc"])

    st.markdown(f"#### {t['p0_metrics_head']}")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label=t["p0_m1_label"],
            value=t["p0_m1_val"],
            delta=t["p0_m1_delta"],
            delta_color="normal"
        )
    with col_m2:
        st.metric(
            label=t["p0_m2_label"],
            value=t["p0_m2_val"],
            delta=t["p0_m2_delta"],
            delta_color="off"
        )
    with col_m3:
        st.metric(
            label=t["p0_m3_label"],
            value=t["p0_m3_val"],
            delta=t["p0_m3_delta"],
            delta_color="off"
        )
    with col_m4:
        st.metric(
            label=t["p0_m4_label"],
            value=t["p0_m4_val"],
            delta=t["p0_m4_delta"],
            delta_color="normal"
        )

    st.markdown("---")

    st.markdown(f"#### {t['p0_seasons_head']}")
    st.caption(t["p0_seasons_caption"])

    tab_kharif, tab_rabi, tab_zaid = st.tabs([
        t["p0_tab_kharif"],
        t["p0_tab_rabi"],
        t["p0_tab_zaid"]
    ])

    with tab_kharif:
        st.markdown(f"""
        <div class="cs-card">
            <h4 style="color:#2e7d32; margin-top:0;">{t['p0_kharif_title']}</h4>
            <p>{t['p0_kharif_p1']}</p>
            <p>{t['p0_kharif_p2']}</p>
            <p>{t['p0_kharif_p3']}</p>
            <div style="background: rgba(46, 125, 50, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #2e7d32; margin-top: 8px;">
                {t['p0_kharif_warn']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_rabi:
        st.markdown(f"""
        <div class="cs-card">
            <h4 style="color:#1565c0; margin-top:0;">{t['p0_rabi_title']}</h4>
            <p>{t['p0_rabi_p1']}</p>
            <p>{t['p0_rabi_p2']}</p>
            <p>{t['p0_rabi_p3']}</p>
            <div style="background: rgba(21, 101, 192, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #1565c0; margin-top: 8px;">
                {t['p0_rabi_warn']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_zaid:
        st.markdown(f"""
        <div class="cs-card">
            <h4 style="color:#e65100; margin-top:0;">{t['p0_zaid_title']}</h4>
            <p>{t['p0_zaid_p1']}</p>
            <p>{t['p0_zaid_p2']}</p>
            <p>{t['p0_zaid_p3']}</p>
            <div style="background: rgba(230, 81, 0, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 4px solid #e65100; margin-top: 8px;">
                {t['p0_zaid_warn']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"#### {t['p0_belts_head']}")
    
    if "selected_region" not in st.session_state or st.session_state["selected_region"] not in AGRO_REGIONS:
        st.session_state["selected_region"] = list(AGRO_REGIONS.keys())[1]

    st.markdown(t["p0_belts_select"])
    b_cols = st.columns(4)
    for i, (r_key, r_val) in enumerate(AGRO_REGIONS.items()):
        with b_cols[i]:
            is_active = (r_key == st.session_state["selected_region"])
            btn_label = r_val.get("short_name_hi" if is_hindi else "short_name", r_key.split("(")[0].strip())
            if st.button(
                btn_label,
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
        st.markdown(get_india_svg(selected_reg_name, is_hindi=is_hindi), unsafe_allow_html=True)

    with col_details:
        reg_title = reg_data["name_hi"] if is_hindi else selected_reg_name
        reg_states = reg_data["states_hi"] if is_hindi else reg_data["states"]
        reg_crops = ', '.join(reg_data["crops_hi"]) if is_hindi else ', '.join([c.replace('_', ' ') for c in reg_data['crops']])
        reg_soil = reg_data["soil_hi"] if is_hindi else reg_data["soil"]
        reg_irr = reg_data["irrigation_hi"] if is_hindi else reg_data["irrigation"]
        reg_clim = reg_data["climate_hi"] if is_hindi else reg_data["climate"]
        reg_threats = reg_data["threats_hi"] if is_hindi else reg_data["threats"]
        reg_why = reg_data["why_hi"] if is_hindi else reg_data["why"]

        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{reg_title}</h3>
            <p><b>{t['p0_card_states']}</b> {reg_states}</p>
            <p><b>{t['p0_card_crops']}</b> <b>{reg_crops}</b></p>
            <p><b>{t['p0_card_soil']}</b> {reg_soil}</p>
            <p><b>{t['p0_card_irrigation']}</b> {reg_irr}</p>
            <p><b>{t['p0_card_climate']}</b> {reg_clim}</p>
            <div style="background: rgba(46, 125, 50, 0.08); padding: 12px; border-radius: 8px; border-left: 4px solid #2e7d32; margin-top: 10px;">
                <b>{t['p0_card_threats']}</b><br/>
                {reg_threats}<br/><br/>
                <b>{t['p0_card_why']}</b> {reg_why}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    cta_col1, cta_col2 = st.columns([1.5, 1.5], gap="medium")
    with cta_col1:
        st.markdown(f"""
        <div class="cs-card" style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
            <div>
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🩺</div>
                <h4 style="margin-top:0; color:#059669; font-weight:700;">{t['p0_cta1_title']}</h4>
                <p style="font-size:0.92rem; opacity:0.85; line-height:1.5;">{t['p0_cta1_desc']}</p>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(128,128,128,0.12); font-size:0.85rem; color:#059669; font-weight:700;">
                {t['p0_cta1_link']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cta_col2:
        st.markdown(f"""
        <div class="cs-card" style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
            <div>
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
                <h4 style="margin-top:0; color:#0284c7; font-weight:700;">{t['p0_cta2_title']}</h4>
                <p style="font-size:0.92rem; opacity:0.85; line-height:1.5;">{t['p0_cta2_desc']}</p>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(128,128,128,0.12); font-size:0.85rem; color:#0284c7; font-weight:700;">
                {t['p0_cta2_link']}
            </div>
        </div>
        """, unsafe_allow_html=True)


elif page == t["nav_diagnose"]:
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

    with st.expander(t["test_guide_title"], expanded=False):
        st.markdown(f"""
        {t['test_guide_1']}  
        {t['test_guide_2']}  
        {t['test_guide_3']}  
        """)

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
                
                st.markdown(f"""
                <div class="cs-result-banner">
                    <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; color:#2e7d32; font-weight:700;">{t['diag_result_title']}</div>
                    <div class="cs-result-title">{info['name']}</div>
                </div>
                """, unsafe_allow_html=True)

                res_left, res_right = st.columns([1, 1], gap="large")

                with res_left:
                    sub_m1, sub_m2 = st.columns(2)
                    sub_m1.metric(t["conf_label"], f"{conf:.1%}")
                    type_display = info["type"].title()
                    if is_hindi:
                        type_map = {"Fungus": "फफूंद", "Bacteria": "जीवाणु", "Virus": "विषाणु", "Pest": "कीट", "Water Mould": "वाटर मोल्ड", "Healthy": "स्वस्थ"}
                        type_display = type_map.get(type_display, type_display)
                    sub_m2.metric(t["type_label"], type_display)

                    if raw_class.endswith("___healthy"):
                        st.success(t["healthy_msg"])

                    i1, i2 = st.columns(2)
                    i1.image(img.resize((224, 224)), caption=t["your_photo"], use_container_width=True)
                    heatmap_overlay_img = Image.fromarray((overlay(img, heat) * 255).astype(np.uint8))
                    i2.image(heatmap_overlay_img, caption=t["heatmap"], use_container_width=True)

                    st.markdown(f"**{t['top3']}**")
                    top3_idx = np.argsort(probs)[::-1][:3]
                    top3_list = []
                    for rank, i in enumerate(top3_idx, start=1):
                        p = float(probs[i])
                        cls_name = remedies[CLASSES[i]]["name"]
                        top3_list.append((cls_name, p))
                        st.progress(p, text=f"{rank}. {cls_name} — {p:.1%}")

                    st.markdown("<br/>", unsafe_allow_html=True)
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

                    healthy_ref_img = get_healthy_reference_image(crop_name)
                    if healthy_ref_img and not raw_class.endswith("___healthy"):
                        with st.expander(t["ref_heading"], expanded=False):
                            c_r1, c_r2 = st.columns(2)
                            crop_display_name = CROP_CATALOG.get(crop_name, {}).get("hindi", crop_name) if is_hindi else crop_name.replace('_', ' ')
                            c_r1.image(healthy_ref_img.resize((200, 200)), caption=t["healthy_cert_caption"].format(crop=crop_display_name))
                            c_r2.image(img.resize((200, 200)), caption=t["submitted_leaf_caption"].format(name=info['name']))


elif page == t["nav_crops"]:
    st.subheader(t["p2_subhead"])
    st.write(t["p2_desc"])

    st.markdown(f"#### {t['p2_sim_head']}")
    st.write(t["p2_sim_desc"])

    sim_c1, sim_c2 = st.columns([1.2, 1.2], gap="large")

    with sim_c1:
        st.markdown(f"##### {t['p2_ctrl_head']}")
        sim_crops_keys = ["Potato", "Tomato", "Apple", "Grape", "Corn_(maize)", "Pepper,_bell", "Orange", "Squash"]
        sim_crop = st.selectbox(
            t["p2_crop_sim_label"],
            sim_crops_keys,
            index=0,
            format_func=lambda x: f"{x.replace('_', ' ')} ({CROP_CATALOG.get(x, {}).get('hindi', x)})" if is_hindi else x.replace('_', ' ')
        )
        sim_temp = st.slider(t["p2_temp_label"], min_value=10, max_value=42, value=18, step=1)
        sim_humidity = st.slider(t["p2_hum_label"], min_value=20, max_value=100, value=85, step=1)
        sim_wet_hours = st.slider(t["p2_wet_label"], min_value=0, max_value=24, value=10, step=1)

    with sim_c2:
        st.markdown(f"##### {t['p2_calc_head']}")
        risk_score = 15
        if is_hindi:
            pathogen_name = "कोई गंभीर रोगज़नक़ नहीं (सामान्य वातावरण)"
            action_advice = "खेत की परिस्थितियां सुरक्षित कृषि मानकों के भीतर हैं। सामान्य नियमित निगरानी जारी रखें।"
        else:
            pathogen_name = "None detected (Normal micro-flora)"
            action_advice = "Field conditions are within safe agronomic parameters. Continue standard routine scouting."
        alert_level = "LOW"

        if sim_crop in ["Potato", "Tomato"]:
            if 12 <= sim_temp <= 22 and sim_humidity >= 80 and sim_wet_hours >= 8:
                risk_score = min(98, 40 + int(sim_wet_hours * 3.5) + int((sim_humidity - 70) * 1.5))
                pathogen_name = "पछेती झुलसा (Phytophthora infestans)" if is_hindi else "Late Blight (Phytophthora infestans)"
                alert_level = "CRITICAL"
                action_advice = "🚨 अत्यधिक गंभीर जोखिम! ठंड और ओस से भरी पत्तियों पर पछेती झुलसा बहुत तेजी से पनपता है। पूरे खेत में फैलने से पहले तुरंत कॉपर युक्त या सिस्टेमिक फफूंदनाशक का छिड़काव करें।" if is_hindi else "🚨 High outbreak risk! Late blight sporangia germinate rapidly in cool, dew-saturated foliage. Apply protective copper or systemic fungicide before full canopy infection."
            elif 22 <= sim_temp <= 30 and sim_humidity >= 70 and sim_wet_hours >= 4:
                risk_score = min(88, 30 + int(sim_wet_hours * 3.0) + int((sim_humidity - 60) * 1.2))
                pathogen_name = "अगेती झुलसा (Alternaria solani)" if is_hindi else "Early Blight (Alternaria solani)"
                alert_level = "MODERATE"
                action_advice = "⚠️ अल्टरनेरिया फफूंद नम और गर्म मौसम में तेजी से फैलती है। पौधे की निचली संक्रमित पत्तियों को तोड़कर नष्ट करें और ऊपर से पानी देने से बचें।" if is_hindi else "⚠️ Alternaria conidia require alternating wet and warm periods. Remove lower infected leaves and avoid overhead irrigation."
            elif sim_temp >= 32 and sim_humidity <= 50:
                risk_score = 72
                pathogen_name = "लाल मकड़ी / सफेद मक्खी (विषाणु वाहक)" if is_hindi else "Spider Mites / Whitefly TYLCV Vector"
                alert_level = "MODERATE"
                action_advice = "गर्म और शुष्क मौसम में मकड़ियों और सफेद मक्खियों का प्रकोप तेजी से बढ़ता है। पत्तियों की निचली सतह की नियमित निगरानी करें।" if is_hindi else "Hot, dry conditions accelerate mite reproduction and whitefly migration. Monitor underside of leaves."

        elif sim_crop in ["Apple", "Cherry"]:
            if 14 <= sim_temp <= 24 and sim_humidity >= 75 and sim_wet_hours >= 6:
                risk_score = min(92, 35 + int(sim_wet_hours * 3.5))
                pathogen_name = "सेब का स्कैब (Venturia inaequalis)" if is_hindi else "Apple Scab (Venturia inaequalis)"
                alert_level = "CRITICAL"
                action_advice = "🚨 पत्तियों पर लंबे समय तक गीलापन रहने से स्कैब के बीजाणु सक्रिय हो गए हैं। 48 घंटों के भीतर सुरक्षात्मक फफूंदनाशक का छिड़काव आवश्यक है।" if is_hindi else "🚨 Mills Period met: Ascospores discharge during prolonged leaf wetness. Fungicidal protection needed within 48 hours."
            elif 18 <= sim_temp <= 28 and 50 <= sim_humidity <= 75 and sim_wet_hours < 4:
                risk_score = 65
                pathogen_name = "चूर्णिल आसिता (Powdery Mildew)" if is_hindi else "Powdery Mildew (Podosphaera leucotricha)"
                alert_level = "MODERATE"
                action_advice = "चूर्णिल आसिता के बीजाणु गर्म और नम हवा में तेजी से फैलते हैं। पेड़ों की भीतरी घनी शाखाओं की छंटाई करें ताकि हवा का संचार बना रहे।" if is_hindi else "Mildew conidia germinate in warm, humid air without free leaf moisture. Prune dense inner canopy for airflow."

        elif sim_crop == "Grape":
            if 20 <= sim_temp <= 30 and sim_humidity >= 75 and sim_wet_hours >= 6:
                risk_score = min(90, 40 + int(sim_wet_hours * 3.2))
                pathogen_name = "काला सड़न रोग (Guignardia bidwellii)" if is_hindi else "Black Rot (Guignardia bidwellii)"
                alert_level = "CRITICAL"
                action_advice = "🚨 गर्म बारिश और पत्तियों पर नमी नई कोपलों और अंगूर के गुच्छों पर काले सड़न के संक्रमण को ट्रिगर करती है।" if is_hindi else "🚨 Warm rain and leaf wetness trigger pycnidiospore infection on young leaves and berry clusters."
            else:
                risk_score = 30

        elif sim_crop == "Corn_(maize)":
            if 18 <= sim_temp <= 28 and sim_humidity >= 80 and sim_wet_hours >= 6:
                risk_score = 78
                pathogen_name = "उत्तरी पत्ता झुलसा (Exserohilum turcicum)" if is_hindi else "Northern Corn Leaf Blight (Exserohilum turcicum)"
                alert_level = "MODERATE"
                action_advice = "भारी ओस और मध्यम तापमान सिगार के आकार के भूरे धब्बों को बढ़ावा देते हैं। फसल चक्र अपनाएं और खेत की स्वच्छता रखें।" if is_hindi else "Heavy dews and moderate temperatures favor cigar-shaped necrotic lesions. Ensure crop rotation and residue management."
            else:
                risk_score = 25

        elif sim_crop == "Pepper,_bell":
            if 24 <= sim_temp <= 32 and sim_humidity >= 80 and sim_wet_hours >= 5:
                risk_score = 84
                pathogen_name = "जीवाणु धब्बा रोग (Xanthomonas campestris)" if is_hindi else "Bacterial Spot (Xanthomonas campestris)"
                alert_level = "CRITICAL"
                action_advice = "तेज हवा, बारिश और गर्मी में जीवाणु रंध्रों (stomata) के जरिए प्रवेश करते हैं। पत्तियां गीली होने पर पौधों को न छुएं।" if is_hindi else "Bacterial pathogens enter via stomata during wind-driven rain and high heat. Avoid handling plants when foliage is wet."
            else:
                risk_score = 30
        else:
            if sim_humidity >= 80 and sim_wet_hours >= 6:
                risk_score = 60
                pathogen_name = "सामान्य फंगल संक्रमण जोखिम" if is_hindi else "General Foliar Fungal Complex"
                alert_level = "MODERATE"
                action_advice = "लगातार नमी से फफूंद पनपने का खतरा है। पत्तियों के किनारों पर नियमित नजर रखें।" if is_hindi else "High moisture creates potential for fungal germination. Monitor leaf margins."

        if alert_level == "CRITICAL":
            st.error(t["p2_risk_critical"].format(risk=risk_score))
        elif alert_level == "MODERATE":
            st.warning(t["p2_risk_mod"].format(risk=risk_score))
        else:
            st.success(t["p2_risk_low"].format(risk=risk_score))

        st.progress(risk_score / 100.0)
        st.markdown(f"**{t['p2_pred_pathogen']}** `{pathogen_name}`")
        st.info(f"**{t['p2_agri_advice']}** {action_advice}")

    st.markdown("---")

    st.markdown(f"#### {t['p2_profile_head']}")
    inspect_crop = st.selectbox(
        t["p2_profile_select"],
        list(CROP_CATALOG.keys()),
        format_func=lambda x: f"{x.replace('_', ' ')} ({CROP_CATALOG[x]['hindi']})"
    )
    c_info = CROP_CATALOG[inspect_crop]

    c_c1, c_c2 = st.columns([1, 2], gap="medium")
    with c_c1:
        c_img = get_healthy_reference_image(inspect_crop)
        crop_disp = c_info['hindi'] if is_hindi else inspect_crop.replace('_', ' ')
        if c_img:
            st.image(get_standardized_image(c_img, (240, 240)), caption=t["p2_profile_caption"].format(crop=crop_disp), use_container_width=True)
        else:
            st.info("प्रमाणित स्वस्थ पत्ती डेटासेट में उपलब्ध" if is_hindi else "Certified Leaf Baseline Available in Dataset")
    with c_c2:
        states_val = c_info['states_hi'] if is_hindi else c_info['states']
        season_val = c_info['season_hi'] if is_hindi else c_info['season']
        soil_val = c_info['soil_hi'] if is_hindi else c_info['soil']
        econ_val = c_info['economic_fact_hi'] if is_hindi else c_info['economic_fact']
        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{inspect_crop.replace('_', ' ')} ({c_info['hindi']})</h3>
            <p><b>{t['p2_profile_states']}</b> {states_val}</p>
            <p><b>{t['p2_profile_season']}</b> {season_val}</p>
            <p><b>{t['p2_profile_soil']}</b> {soil_val}</p>
            <p><b>{t['p2_profile_econ']}</b> {econ_val}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"#### {t['p2_conditions_head'].format(crop=c_info['hindi'] if is_hindi else inspect_crop.replace('_', ' '))}")
    st.caption(t["p2_conditions_caption"])
    
    prefix = "Cherry_(including_sour)___" if inspect_crop == "Cherry" else f"{inspect_crop}___"
    crop_classes = [c for c in CLASSES if c.startswith(prefix)]
    
    ext_classes = [k for k in EXTENDED_PATHOLOGY if k.startswith(f"{inspect_crop}___") and k not in crop_classes]
    all_display_classes = crop_classes + ext_classes

    cols = st.columns(len(all_display_classes) if len(all_display_classes) <= 3 else 3)
    for idx, c in enumerate(all_display_classes):
        with cols[idx % 3]:
            if c in EXTENDED_PATHOLOGY:
                ext_item = EXTENDED_PATHOLOGY[c]
                sample_img_path = ext_item["image"]
                cond_name = ext_item["name_hi"] if is_hindi else ext_item["name"]
                cond_type = ext_item["type_hi"] if is_hindi else ext_item["type"].title()
                cond_symptoms = ext_item["symptoms_hi"] if is_hindi else ext_item["symptoms"]
                cond_prev = ext_item["prevention_hi"][0] if is_hindi else ext_item["prevention"][0]
            else:
                info = remedies[c]
                sample_img_path = Path("demo_images") / f"{c}.jpg"
                if not sample_img_path.exists() and "Cherry" in c:
                    sample_img_path = Path("demo_images") / "Cherry_(including_sour)___healthy.jpg"
                cond_name = info['name']
                cond_type = info['type'].title()
                if is_hindi:
                    type_map = {"Fungus": "फफूंद", "Bacteria": "जीवाणु", "Virus": "विषाणु", "Pest": "कीट", "Water Mould": "वाटर मोल्ड", "Healthy": "स्वस्थ"}
                    cond_type = type_map.get(cond_type, cond_type)
                cond_symptoms = info['symptoms']
                cond_prev = info['prevention'][0] if info.get('prevention') else ("खेत की स्वच्छता व हवा का संचार बनाए रखें" if is_hindi else "Standard field sanitation and air circulation")
            
            fitted_img = get_standardized_image(sample_img_path, (240, 240))
            if fitted_img:
                st.image(fitted_img, caption=f"🔬 {cond_name}", use_container_width=True)
            
            st.markdown(f"""
            <div class="cs-card" style="margin-top: -6px; margin-bottom: 20px;">
                <h4 style="margin-top:0; color:#2e7d32;">{cond_name}</h4>
                <p><b>{t['p2_card_type']}</b> <span style="background:rgba(46, 125, 50, 0.12); padding: 2px 8px; border-radius: 4px; font-weight:600;">{cond_type}</span></p>
                <p style="font-size: 0.9rem;"><b>{t['p2_card_symptoms']}</b><br/>{cond_symptoms}</p>
                <div style="background: rgba(46, 125, 50, 0.08); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #2e7d32; margin-top: 6px; font-size: 0.88rem;">
                    <b>{t['p2_card_prev']}</b><br/>
                    {cond_prev}
                </div>
            </div>
            """, unsafe_allow_html=True)


elif page == t["nav_kvk"]:
    st.subheader(t["p3_subhead"])
    st.write(t["p3_desc"])

    state_choice = st.selectbox(t["select_state"], list(kvk_directory.keys()))
    kvk_selected = kvk_directory[state_choice]

    col_kvk_card, _ = st.columns([2, 1])
    with col_kvk_card:
        st.markdown(f"""
        <div class="cs-card">
            <h3 style="margin-top:0; color:#2e7d32;">{kvk_selected['kvk']}</h3>
            <p><b>{t['p3_districts']}</b> {kvk_selected['district']}</p>
            <p><b>{t['p3_contact']}</b> {kvk_selected['contact']}</p>
            <p style="background: rgba(46, 125, 50, 0.12); padding: 12px; border-radius: 8px;">
                <b>{t['p3_kcc']}</b> <span style="font-size:1.1rem; font-weight:700; color:#b71c1c;">{kvk_selected['helpline']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
