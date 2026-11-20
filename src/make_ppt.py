from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent.parent
REP, DEMO = ROOT / "reports", ROOT / "demo_images"
GREEN, DARK, LIGHT, WHITE = RGBColor(0x2E, 0x7D, 0x32), RGBColor(0x1B, 0x3A, 0x1E), RGBColor(0xF1, 0xF8, 0xE9), RGBColor(255, 255, 255)
GREY = RGBColor(0x45, 0x55, 0x45)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rect(s, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = color; r.line.fill.background()
    return r


def text(s, x, y, w, h, t, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tb


def bullets(s, x, y, w, h, items, size=20):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        r = p.add_run(); r.text = "•  " + it; r.font.size = Pt(size); r.font.color.rgb = GREY


def picture(s, path, x, y, w, h):
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def slide(title):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 13.333, 1.2, GREEN)
    text(s, 0.6, 0.25, 12, 0.8, title, 32, True, WHITE)
    rect(s, 0, 7.2, 13.333, 0.3, DARK)
    return s


def table(s, x, y, w, rows, colw=None, size=15):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.5 * len(rows))).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.font.bold = (i == 0)
                    r.font.color.rgb = WHITE if i == 0 else DARK
            c.fill.solid(); c.fill.fore_color.rgb = GREEN if i == 0 else (LIGHT if i % 2 else WHITE)
    if colw:
        for j, cw in enumerate(colw):
            t.columns[j].width = Inches(cw)


s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, DARK)
rect(s, 0, 4.9, 13.333, 0.08, GREEN)
text(s, 0.8, 2.0, 11.5, 1.2, "🌿 CropShield", 66, True, WHITE)
text(s, 0.8, 3.4, 11.5, 1, "Crop Disease & Pest Classifier using Deep Learning", 30, False, RGBColor(0xC8, 0xE6, 0xC9))
text(s, 0.8, 5.2, 11.5, 0.6, "Leaf photo  →  Diagnosis  →  Explanation  →  Treatment advice", 20, False, WHITE)
text(s, 0.8, 6.3, 11.5, 0.5, "Class Presentation", 16, False, RGBColor(0xA5, 0xD6, 0xA7))

s = slide("The Problem")
bullets(s, 0.7, 1.7, 7.2, 5, [
    "Crop diseases and pests cause major yield losses for farmers every year",
    "Farmers often can't reach an expert quickly; late diagnosis means wrong or delayed treatment",
    "Many diseases look alike to the naked eye (e.g. fungal vs. pest damage)",
    "Goal: a phone-friendly tool that diagnoses a leaf photo in seconds and advises what to do",
], 22)
picture(s, REP / "samples.png", 8.1, 1.7, 4.8, 5.0)

s = slide("Our Solution – 4 Steps")
steps = [("1. Leaf Check", "Rejects non-leaf photos\n(selfies, objects)"),
         ("2. Classify", "ResNet18 identifies crop + disease\n(38 classes, 14 crops)"),
         ("3. Explain", "Grad-CAM heatmap shows where\nthe model looked"),
         ("4. Advise", "Bilingual (English / हिन्दी)\ntreatment & prevention")]
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * 3.15
    rect(s, x, 2.0, 2.9, 3.6, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, x, 2.0, 2.9, 0.9, GREEN, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 2.15, 2.9, 0.6, h, 24, True, WHITE, PP_ALIGN.CENTER)
    text(s, x + 0.15, 3.3, 2.6, 2, d, 18, False, GREY, PP_ALIGN.CENTER)
text(s, 0.6, 6.0, 12, 0.6, "Safety built in: 80% confidence threshold + 'consult your local KVK / agriculture officer' disclaimer", 18, True, GREEN)

s = slide("Datasets Used")
table(s, 0.7, 1.7, 11.9, [
    ["Dataset", "Type", "Size", "Purpose"],
    ["PlantVillage", "Studio photos, plain background", "54,305 images, 38 classes", "Main training & benchmark (80/10/10 split)"],
    ["PlantDoc", "Real field photos", "2,575 images, 28 classes mapped", "Field testing & domain adaptation"],
    ["Non-leaf photos", "People, buildings, vehicles…", "201 test images", "Train/test the 'Not a Leaf' detector"],
], [2.4, 3.2, 3.0, 3.3], 16)
bullets(s, 0.7, 4.2, 11.9, 2.8, [
    "14 crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato",
    "Studio images are easy – real fields have shadows, clutter and mixed lighting (domain shift)",
], 19)

s = slide("Models: Baseline vs Transfer Learning")
table(s, 0.7, 1.7, 11.9, [
    ["", "SimpleCNN (from scratch)", "ResNet18 (ImageNet pretrained)"],
    ["Architecture", "4 conv blocks + global pool + dropout", "ResNet18, final layer → 38 classes"],
    ["Parameters", "399,142", "11,196,006"],
    ["Training", "8 epochs, Adam 1e-3", "5 epochs, Adam 3e-4, cosine annealing"],
    ["Test accuracy", "95.47%", "99.69%"],
    ["Mistakes (5,431 images)", "246", "17"],
    ["Macro F1", "0.942", "0.996"],
], [3.2, 4.2, 4.5], 17)
text(s, 0.7, 5.7, 11.9, 1, "Transfer learning gives ~14× fewer errors with fewer epochs – pretrained features already know edges, textures and shapes.", 19, True, GREEN)

s = slide("Where Does the Model Make Mistakes?")
picture(s, REP / "resnet18_confusion_matrix.png", 0.5, 1.4, 6.6, 5.7)
bullets(s, 7.3, 1.6, 5.6, 5.5, [
    "13 of 17 errors stay within the same crop – the plant is recognised, similar diseases get confused",
    "Corn Cercospora ↔ Northern Leaf Blight (both grey-tan streaks)",
    "Tomato Target Spot ↔ Spider mites (tiny speckles)",
    "Some errors come from bad dataset images (e.g. 'Cherry healthy' showing grass)",
], 18)

s = slide("Explainability: Grad-CAM")
picture(s, REP / "gradcam_correct.png", 0.5, 1.4, 7.4, 5.7)
bullets(s, 8.1, 1.6, 4.9, 5.5, [
    "Heatmap on ResNet18's last conv block, built from scratch with hooks",
    "Diseased leaf → attention on lesions and discoloured patches",
    "Healthy leaf → attention on the leaf outline",
    "Builds trust: shows the model uses disease features, not the background",
], 18)

s = slide("Lab vs Real Field: Closing the Gap")
table(s, 0.7, 1.7, 11.9, [
    ["PlantDoc field test (238 images)", "SimpleCNN", "ResNet18 (lab-only)", "ResNet18 (field-adapted)"],
    ["Top-1 accuracy", "10.08%", "20.17%", "61.34%"],
    ["Top-3 accuracy", "25.63%", "43.28%", "87.39%"],
    ["Crop-level accuracy", "26.89%", "46.64%", "79.41%"],
    ["PlantVillage (lab) accuracy", "95.47%", "99.69%", "98.47%"],
], [4.2, 2.3, 2.7, 2.7], 17)
bullets(s, 0.7, 4.5, 11.9, 2.5, [
    "Lab-only models collapse on real photos (99.7% → 20%)",
    "Fix: strong field-style augmentation + PlantDoc training split (test images never used for training)",
    "Result: +41 percentage points on field data with only −1.2 pp on lab data",
], 20)

s = slide("'Not a Leaf' Detector")
table(s, 0.7, 1.7, 11.9, [
    ["Metric (739 unseen images)", "Naive 80% confidence gate", "Dedicated LeafDetector"],
    ["Overall accuracy", "76.73%", "99.05%"],
    ["Real field leaves accepted", "52.94%", "99.16%"],
    ["Non-leaf rejection rate", "69.65%", "98.01%"],
    ["Non-leaf false acceptance", "30.35%", "1.99%"],
], [4.5, 3.7, 3.7], 17)
bullets(s, 0.7, 4.5, 11.9, 2.5, [
    "A softmax classifier must pick one of 38 diseases even for a selfie",
    "Solution: frozen ImageNet ResNet18 backbone + logistic regression head (only 4 KB), ROC-AUC 0.9998",
], 20)

s = slide("The CropShield Web App (Streamlit)")
pages = [("🌾 Indian Agriculture", "Vector map of India: farming belts, regions and their crops"),
         ("🩺 Leaf Diagnosis", "Upload / camera photo → prediction, confidence, Grad-CAM, remedies"),
         ("🔬 Crop Pathology", "Healthy vs diseased reference for all 14 crops + outbreak simulator"),
         ("🏛️ KVK & Helplines", "State-wise Krishi Vigyan Kendra contacts and farmer helplines")]
for i, (h, d) in enumerate(pages):
    x = 0.6 + (i % 2) * 6.3
    y = 1.6 + (i // 2) * 1.9
    rect(s, x, y, 6.0, 1.7, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x + 0.2, y + 0.1, 5.6, 0.5, h, 22, True, GREEN)
    text(s, x + 0.2, y + 0.7, 5.6, 1, d, 16, False, GREY)
imgs = [DEMO / "Apple___Apple_scab.jpg", DEMO / "Corn_(maize)___Common_rust_.jpg", DEMO / "Grape___Black_rot.jpg", DEMO / "Orange___healthy.jpg"]
for i, p in enumerate(imgs):
    if p.exists():
        picture(s, p, 0.8 + i * 3.1, 5.4, 2.8, 1.7)

s = slide("Limitations & Future Work")
bullets(s, 0.7, 1.7, 6, 5, [
    "Field accuracy (61% top-1) is still well below studio accuracy",
    "Similar-looking diseases are confused; one pest class only",
    "Heatmap resolution is 7×7 – regions, not exact lesion outlines",
    "Dataset has label noise",
], 20)
bullets(s, 7.0, 1.7, 5.8, 5, [
    "Collect more real field images across Indian crops",
    "Bigger models / ensembles, severity estimation",
    "Offline mobile app and more regional languages",
    "Link with KVK experts for human confirmation",
], 20)
text(s, 0.7, 1.3, 6, 0.4, "Limitations", 16, True, GREEN)
text(s, 7.0, 1.3, 6, 0.4, "Future Work", 16, True, GREEN)

s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, DARK)
text(s, 0.8, 0.8, 11.5, 1, "Conclusion", 44, True, WHITE)
bullets(s, 0.8, 2.2, 11.7, 4, [
    "End-to-end pipeline: leaf check → diagnosis → explanation → bilingual advice",
    "ResNet18 transfer learning: 99.69% on PlantVillage, 61.34% top-1 / 87.39% top-3 on real field photos",
    "Explainable and safety-aware: Grad-CAM, confidence threshold, KVK referral",
], 24)
for p in s.shapes:
    if p.has_text_frame and p.text_frame.text.startswith("•"):
        for para in p.text_frame.paragraphs:
            for r in para.runs:
                r.font.color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
text(s, 0.8, 6.0, 11.5, 0.8, "Thank you!  Questions?", 36, True, RGBColor(0xA5, 0xD6, 0xA7))

out = ROOT / "CropShield_Presentation.pptx"
prs.save(out)
print("saved", out)
