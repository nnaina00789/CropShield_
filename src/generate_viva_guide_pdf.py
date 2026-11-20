"""
Generates the comprehensive CropShield Viva & Project Master Guide PDF.
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
PDF_OUT = ROOT / "CropShield_Viva_and_Project_Guide.pdf"

class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        
        # Header (on pages after page 1)
        if self._pageNumber > 1:
            self.drawString(40, 755, "CropShield: Crop Disease & Pest Classifier — Master Project & Viva Guide")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 28, page_str)
        self.drawString(40, 28, "Confidential — Academic Presentation & Viva Preparation Document")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(40, 38, 572, 38)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1b5e20'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#37474f'),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1b5e20'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2e7d32'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#212121'),
        spaceAfter=6
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    bullet_style = ParagraphStyle(
        'BulletItem',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )
    q_style = ParagraphStyle(
        'QuestionStyle',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0d5302'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    a_style = ParagraphStyle(
        'AnswerStyle',
        parent=body_style,
        leftIndent=8,
        spaceAfter=6
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1b5e20')
    )

    story = []

    # Title Banner
    story.append(Paragraph("🌿 CropShield: Deep Learning Leaf Disease Diagnosis", title_style))
    story.append(Paragraph("<b>End-to-End System Architecture, Machine Learning Theory & Comprehensive Viva Guide</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2e7d32"), spaceAfter=12))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Statement", h1_style))
    story.append(Paragraph(
        "Crop diseases and insect pests cause between <b>20% to 40% of global agricultural yield loss</b> annually. "
        "In developing farming regions like India, smallholder farmers often lack immediate access to certified plant pathologists. "
        "Furthermore, many foliar diseases appear identical to the naked eye (such as bacterial leaf spot vs. fungal early blight, "
        "or spider mite speckling vs. target spot). Misdiagnosing a pathogen leads farmers to spray improper chemicals (e.g., applying fungicides "
        "for viral infections), which wastes economic resources, damages soil health, and allows the epidemic to spread unchecked.",
        body_style
    ))
    story.append(Paragraph(
        "<b>CropShield</b> is an end-to-end computer vision and agricultural decision-support system. It processes a leaf photograph in seconds, "
        "filters out non-leaf photos using an Out-of-Distribution gatekeeper, classifies the condition across 38 crop-disease categories (14 crops) "
        "using transfer learning, visually explains its decision via Grad-CAM heatmaps, and provides bilingual treatment recommendations with safety disclaimers.",
        body_style
    ))

    # Section 2: Datasets & Explanations
    story.append(Paragraph("2. Datasets Used: The Numbers & Rationale", h1_style))
    story.append(Paragraph(
        "Our project leverages three distinct image datasets to ensure both lab precision and real-world robustness:",
        body_style
    ))

    ds_data = [
        [Paragraph("<b>Dataset</b>", body_bold), Paragraph("<b>Nature of Images</b>", body_bold), Paragraph("<b>Count</b>", body_bold), Paragraph("<b>Project Role</b>", body_bold)],
        [Paragraph("PlantVillage", body_style), Paragraph("Clean studio leaf photos, uniform grey/black backgrounds, controlled lighting", body_style), Paragraph("54,305", body_style), Paragraph("Primary training & lab benchmark (80/10/10 stratified split)", body_style)],
        [Paragraph("PlantDoc", body_style), Paragraph("Real-world field photos, outdoor sunlight, soil, weeds, multiple leaves", body_style), Paragraph("2,575", body_style), Paragraph("Outdoor field testing & domain adaptation fine-tuning", body_style)],
        [Paragraph("Non-Leaf OOD", body_style), Paragraph("People, vehicles, animals, indoor objects, buildings", body_style), Paragraph("401", body_style), Paragraph("Training & testing the Leaf vs. Non-Leaf Gatekeeper", body_style)],
        [Paragraph("<b>Total Images</b>", body_bold), Paragraph("<b>Combined Dataset Scale</b>", body_bold), Paragraph("<b>57,281</b>", body_bold), Paragraph("<b>Complete Multitask Pipeline</b>", body_bold)]
    ]
    t_ds = Table(ds_data, colWidths=[80, 190, 60, 202])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8f5e9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Why only 14 crops?</b>", h2_style))
    story.append(Paragraph(
        "PlantVillage is the established, peer-reviewed global standard benchmark in foliar disease classification. "
        "It focuses on 14 economically vital agricultural and horticultural crops: <i>Apple, Blueberry, Cherry, Corn (Maize), Grape, "
        "Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato</i>. These represent multiple biological families "
        "(Solanaceae, Rosaceae, Poaceae, Fabaceae, Cucurbitaceae, Rutaceae, Vitaceae). In academic research, using standardized benchmarks is essential "
        "so that accuracy and F1 scores can be objectively compared against published literature.",
        body_style
    ))

    story.append(Paragraph("<b>Why are there 44 demo images in the project folder?</b>", h2_style))
    story.append(Paragraph(
        "The original PlantVillage benchmark had 38 classes, but it contained a curation asymmetry: some crops only had healthy controls "
        "(Soybean, Blueberry, Raspberry), while Orange and Squash only had diseased epidemic samples! In a production app, evaluators and farmers "
        "need to see what both a healthy and a diseased leaf look like for every crop. We curated <b>38 original benchmark images + 6 verified field "
        "reference images = 44 demo images</b> (including authentic Citrus leaf and Cucurbit leaf images) so that all 14 crops have complete 360° photographic references.",
        body_style
    ))

    story.append(Paragraph("<b>What does '38 Classes' mean?</b>", h2_style))
    story.append(Paragraph(
        "In computer vision, a 'class' is the label the network predicts. CropShield predicts a compound label: <b>Crop + Health Condition</b>. "
        "For example, <i>'Tomato___Early_blight'</i> or <i>'Apple___healthy'</i>. Across the 14 crops, there are exactly 38 specific categories (26 disease states + 12 healthy controls). "
        "<i>(Note: This refers to 38 pathological classes, not the 28 states/UTs of India).</i>",
        body_style
    ))

    story.append(Paragraph("<b>Where do the Non-Leaf Out-of-Distribution (OOD) photos come from?</b>", h2_style))
    story.append(Paragraph(
        "The 401 non-leaf images in <code>data/raw/non_leaf/</code> were curated from two verified open-access sources: "
        "<br/>1. <b>Unsplash Photography via Lorem Picsum (picsum.photos):</b> 400 diverse high-resolution photographs spanning human portraits, "
        "urban architecture, automobiles, household furniture, electronic gadgets, landscapes, and animals. "
        "<br/>2. <b>Scikit-Learn Built-in Reference Images:</b> Architectural photography (<code>china_palace.jpg</code>). "
        "<br/>This comprehensive diversity across everyday non-agricultural scenes ensures the LeafDetector Gatekeeper learns a robust, generalizable boundary between plant foliage and non-leaf objects.",
        body_style
    ))

    # Section 3: Machine Learning Concepts
    story.append(Paragraph("3. Core Machine Learning Concepts Simplified", h1_style))

    story.append(Paragraph("<b>1. What are 'Weights' in a Neural Network?</b>", h2_style))
    story.append(Paragraph(
        "Think of a mathematical formula: <i>Disease Score = (w1 × brown lesion) + (w2 × yellow border) + (w3 × leaf vein)</i>. "
        "The multiplier values (<b>w1, w2, w3</b>) are the <b>weights</b>. They represent the strength of connection between artificial neurons. "
        "At the start of training, weights are initialized to random numbers (the model knows nothing). During training, the network predicts labels, "
        "measures its mistake (Loss), and uses <b>Backpropagation</b> to tweak every weight so it makes fewer mistakes next time. "
        "Our ResNet-18 model contains <b>11.2 million weights</b> stored in <code>checkpoints/resnet18.pt</code>.",
        body_style
    ))

    story.append(Paragraph("<b>2. What is an 'Epoch'?</b>", h2_style))
    story.append(Paragraph(
        "<b>One Epoch</b> means the neural network has processed <b>every single training image in the dataset exactly once</b>. "
        "Because our training set contains 43,444 images, passing them in mini-batches of 32 takes 1,357 iterations per epoch. "
        "Training for 5 epochs means the model studied the entire 43,444-image dataset 5 full times. We stop at 5 epochs because the validation accuracy "
        "already reached 99.7%; training further risks <b>overfitting</b> (memorizing noise instead of general patterns).",
        body_style
    ))

    story.append(Paragraph("<b>3. What does 'Stratified Split' mean? (The Marble Analogy)</b>", h2_style))
    story.append(Paragraph(
        "Imagine a jar with 100 marbles: 80 Red (common disease), 15 Blue (rare disease), and 5 Gold (very rare disease). "
        "If you randomly scoop 10 marbles, you might get zero Gold marbles by pure accident! Your test set would fail to test Gold marbles completely.<br/>"
        "A <b>Stratified Split</b> enforces that every subset maintains the <b>exact same percentage ratio</b> as the whole. "
        "In CropShield's 80/10/10 split, Orange Huanglongbing (5,507 images) and Potato Healthy (152 images) both have exactly 80% in train, 10% in validation, and 10% in test. "
        "This guarantees rare classes are tested fairly.",
        body_style
    ))

    # Section 4: Models & Architecture
    story.append(Paragraph("4. Model Architectures: How Each Model Works", h1_style))

    models_data = [
        [Paragraph("<b>Model</b>", body_bold), Paragraph("<b>Parameters</b>", body_bold), Paragraph("<b>Architecture & Working Principle</b>", body_bold), Paragraph("<b>Role & Performance</b>", body_bold)],
        [
            Paragraph("<b>SimpleCNN</b>", body_style),
            Paragraph("399,142<br/>(~400K)", body_style),
            Paragraph("4 Convolutional blocks (Conv2D $\\to$ BatchNorm $\\to$ ReLU $\\to$ MaxPool), filter depth 32$\\to$64$\\to$128$\\to$256, Global Average Pooling, Dropout (0.3), Linear head.", body_style),
            Paragraph("<b>Baseline from scratch.</b> Achieves 95.47% test acc, but struggles on rare classes (Potato healthy recall 80%, Tomato Early blight recall 72%).", body_style)
        ],
        [
            Paragraph("<b>ResNet-18</b><br/>(Transfer Learning)", body_style),
            Paragraph("11,196,006<br/>(~11.2M)", body_style),
            Paragraph("Pretrained on ImageNet (1.4M images). Uses <b>Residual Skip Connections ($F(x) + x$)</b> that bypass layers, preventing vanishing gradients. Final linear layer adapted to 38 classes.", body_style),
            Paragraph("<b>Primary Classifier.</b> <b>99.69% test acc</b> (only 17 errors in 5,431 images). 14× fewer errors than baseline.", body_style)
        ],
        [
            Paragraph("<b>Field-Adapted ResNet-18</b>", body_style),
            Paragraph("11,196,006", body_style),
            Paragraph("Fine-tuned ResNet-18 on 3,800 PlantVillage images (100/class) + 4,674 PlantDoc field images (2,337, 2× oversampled) = 8,474 images/epoch, with aggressive outdoor augmentations (color jitter, perspective tilt, Gaussian blur).", body_style),
            Paragraph("<b>Outdoor Deployable.</b> Boosts PlantDoc field accuracy from 20.17% to <b>61.34% Top-1</b> and <b>87.39% Top-3</b> (+41 pp gain).", body_style)
        ],
        [
            Paragraph("<b>LeafDetector Gatekeeper</b>", body_style),
            Paragraph("513 weights<br/>(<b>4 KB</b> file)", body_style),
            Paragraph("Frozen ResNet-18 feature extractor (512-dim embedding) + L2-regularized Logistic Regression hyperplane classifier.", body_style),
            Paragraph("<b>Security Bouncer.</b> Filters out non-leaf photos (selfies, cars) with <b>99.05% accuracy</b> and 0.9998 ROC-AUC.", body_style)
        ]
    ]
    t_models = Table(models_data, colWidths=[70, 70, 210, 182])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8f5e9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 10))

    # Section 5: The Gatekeeper Bouncer Analogy
    story.append(Paragraph("5. The Leaf Gatekeeper: The 'Nightclub Bouncer' Analogy", h1_style))
    story.append(Paragraph(
        "Standard image classification networks use a mathematical function called <b>Softmax</b> in their final layer. "
        "Softmax takes raw outputs and forces them to sum to <b>100%</b> across the 38 plant classes. "
        "Because of this, if an end-user takes a picture of their face, their hand, or a tractor, the network <i>cannot</i> say 'None of the above'. "
        "It will forcefully pick one plant disease and confidently announce: <i>'You are 96% Tomato Early Blight'</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Nightclub Bouncer Solution:</b><br/>"
        "Think of CropShield as an exclusive clinic where 38 specialist plant doctors work. "
        "The <b>Leaf Gatekeeper is the bouncer stationed at the front door</b>. "
        "Before any uploaded image is allowed to see the 38 disease doctors, the bouncer inspects it and asks: <i>'Are you an actual plant leaf?'</i><br/>"
        "• If you upload a selfie or a dog, the bouncer stops you immediately with a red alert: <i>'Invalid image: No crop foliage detected.'</i><br/>"
        "• If you upload an authentic leaf, the bouncer lets you inside to receive diagnosis from ResNet-18.<br/>"
        "<b>Why not just use an 80% confidence threshold?</b> We tested that! On 201 non-leaf photos, an 80% threshold still allowed <b>30.35% of non-leaf photos to sneak through</b>, "
        "while wrongly rejecting <b>47% of real field leaves</b>. Our dedicated Gatekeeper accepts 99.16% of real field leaves and blocks 98.01% of non-leaf photos.",
        body_style
    ))

    # Section 6: Results & Evaluation
    story.append(Paragraph("6. Results & Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "<b>Summary on 5,431 Unseen Test Images (PlantVillage):</b>",
        body_style
    ))

    res_data = [
        [Paragraph("<b>Metric</b>", body_bold), Paragraph("<b>SimpleCNN (Baseline)</b>", body_bold), Paragraph("<b>ResNet-18 (CropShield)</b>", body_bold), Paragraph("<b>Significance</b>", body_bold)],
        [Paragraph("Test Accuracy", body_style), Paragraph("95.47%", body_style), Paragraph("<b>99.69%</b>", body_bold), Paragraph("Only 17 errors out of 5,431 images", body_style)],
        [Paragraph("Macro Precision", body_style), Paragraph("0.945", body_style), Paragraph("<b>0.997</b> (99.7%)", body_bold), Paragraph("Extremely low false alarm rate", body_style)],
        [Paragraph("Macro Recall", body_style), Paragraph("0.941", body_style), Paragraph("<b>0.996</b> (99.6%)", body_bold), Paragraph("Virtually zero missed infections", body_style)],
        [Paragraph("Macro F1-Score", body_style), Paragraph("0.942", body_style), Paragraph("<b>0.996</b>", body_bold), Paragraph("Equally strong on rare & dominant classes", body_style)],
        [Paragraph("Worst Class Recall", body_style), Paragraph("72.0% (Tomato Early blight)", body_style), Paragraph("<b>94.1%</b> (Corn Cercospora)", body_bold), Paragraph("Eliminated blind spots on hard diseases", body_style)]
    ]
    t_res = Table(res_data, colWidths=[100, 110, 110, 212])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8f5e9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Precision vs. Recall in Agriculture: Which is More Dangerous?</b><br/>"
        "• <b>Precision:</b> When the AI says a leaf has <i>Potato Late Blight</i>, is it really Late Blight? (Low precision = unnecessary chemical spraying).<br/>"
        "• <b>Recall:</b> Out of 100 diseased leaves in the field, did the AI catch all 100? (Low recall = missed infections).<br/>"
        "<b>Recall is far more dangerous to get wrong.</b> If an aggressive fungus like Late Blight is misdiagnosed as 'Healthy' (a False Negative), "
        "the farmer takes no action, and within 7 to 10 days the pathogen can ruin the entire crop. CropShield achieves 99.6% recall.",
        body_style
    ))

    # Section 7: Explainable AI: Grad-CAM
    story.append(Paragraph("7. Explainable AI: Grad-CAM Working & Need", h1_style))
    story.append(Paragraph(
        "<b>The Need:</b> Deep learning models are often criticized as 'Black Boxes'. If a model tells a farmer to discard a crop, "
        "the farmer needs proof that the AI looked at the actual disease spots rather than a shadow or leaf shape.<br/>"
        "<b>How Grad-CAM Works:</b><br/>"
        "1. When an image passes forward through ResNet-18, the last convolutional layer (<code>layer4</code>) produces 512 feature maps of size 7×7.<br/>"
        "2. We take the gradient (derivative) of the predicted disease score with respect to these 512 feature maps.<br/>"
        "3. We calculate an importance weight for each feature map: feature maps that fired strongly on the disease get higher weights.<br/>"
        "4. We combine them and pass through a <b>ReLU</b> activation (discarding negative pixels) to produce a 2D intensity heatmap.<br/>"
        "5. The heatmap is upscaled to 224×224 and blended over the leaf photo as a color overlay (Red = intense focus, Blue = ignored).<br/>"
        "<b>What it proved in CropShield:</b> For diseased leaves, the heat spots lock directly onto necrotic lesions and pustules. "
        "For healthy leaves, the attention frames the leaf outline to identify the crop species.",
        body_style
    ))

    # Section 8: Web Application
    story.append(Paragraph("8. The Web Application Portal (Streamlit)", h1_style))
    story.append(Paragraph(
        "The web application (<code>app/app.py</code>) provides 4 clean, production-ready modules with zero external server dependencies:<br/>"
        "1. <b>🌾 Indian Agriculture & Farming Belts:</b> Features a dedicated vector SVG map of India rendered inline. Clicking or selecting a region details "
        "soil types, climate characteristics, and the specific dataset crops cultivated in each state.<br/>"
        "2. <b>🩺 Leaf Diagnosis:</b> Dual-mode inference (Standard Studio Mode & Field-Adapted Mode). Supports file upload or live camera capture, "
        "OOD gatekeeper screening, Grad-CAM side-by-side heatmaps, and bilingual treatments (English & Hindi).<br/>"
        "3. <b>🔬 Crop Pathology & Outbreak Simulator:</b> Reference catalog of healthy vs. diseased leaves for all 14 crops, plus an epidemiology simulator "
        "using Wallin & Smith micro-climate criteria (Temperature, Humidity, Leaf Wetness Duration) to calculate real-time fungal outbreak risks.<br/>"
        "4. <b>🏛️ KVK & Farmer Helplines:</b> State-wise directory of Krishi Vigyan Kendras with direct advisory hotlines.",
        body_style
    ))

    # Section 9: Top 10 Viva Questions & Answers
    story.append(Paragraph("9. Top 10 Viva Questions & Winning Answers", h1_style))

    viva_qa = [
        ("Q1: Why did you choose ResNet-18 instead of a deeper model like ResNet-50 or ViT?",
         "ResNet-18 provides the optimal balance of high accuracy (99.69%), low memory footprint (44 MB), and fast inference latency (~15 ms on CPU). For deployment in rural agricultural centers or edge devices, ResNet-50 (25.6M params) or Vision Transformers (86M+ params) triple latency and storage costs with negligible accuracy gains over 99.69%."),
        ("Q2: Why did lab accuracy (99.7%) drop to 20.1% on field photos, and how did you fix it?",
         "This is known as Domain Shift (or Covariate Shift). PlantVillage photos have clean backgrounds and studio lighting, whereas PlantDoc field photos have soil, direct sunlight, and complex foliage. The lab model learned spurious background shortcuts. We solved it by fine-tuning ResNet-18 with aggressive field augmentations (color jitter, perspective tilt) and PlantDoc training images, boosting outdoor accuracy to 61.34% Top-1 and 87.39% Top-3."),
        ("Q3: What happens if an end-user uploads a picture of a car or a human face?",
         "Standard softmax classifiers are closed-set and would hallucinate a confident plant disease. To prevent this, we built a dedicated LeafDetector Gatekeeper (frozen ResNet backbone + logistic regression head, 4 KB). It screens out non-leaf photos with 99.05% binary accuracy and a 0.9998 ROC-AUC before disease inference runs."),
        ("Q4: What is the purpose of Grad-CAM in your application?",
         "Grad-CAM (Gradient-weighted Class Activation Mapping) solves the 'Black Box' problem. It computes the gradients of the winning class score with respect to layer4 feature maps, generating a heatmap that shows exactly which leaf lesions drove the decision. This establishes clinical and agricultural trust for farmers."),
        ("Q5: Why is Macro F1-score better than Accuracy for this dataset?",
         "PlantVillage exhibits class imbalance (e.g., 5,507 Orange Citrus Greening images vs. only 152 Potato Healthy images). Overall accuracy is biased toward the majority class. Macro F1 computes the F1 score for each of the 38 classes independently and averages them equally, ensuring rare diseases are diagnosed just as reliably as common ones."),
        ("Q6: What is the difference between an Epoch and a Batch?",
         "A batch is a small subset of images (e.g., 32 images) processed in one forward/backward pass. An epoch is completed only when the model has seen every single image in the entire training dataset (1,357 batches = 43,444 images = 1 epoch)."),
        ("Q7: Why do your treatment recommendations classify diseases by pathogen type?",
         "Because agrochemicals cannot cross biological kingdoms. Fungicides have zero efficacy against bacteria or viruses (like Tomato Yellow Leaf Curl or Mosaic Virus), which require insect vector control and physical roguing. Giving pathogen-specific advice prevents farmers from misapplying expensive and environmentally damaging chemicals."),
        ("Q8: Why does your model have an 80% confidence threshold?",
         "In agricultural safety, showing 'I am not sure, please consult an agronomist' is vastly safer than guessing incorrectly on ambiguous leaf photos. If the highest probability is below 80%, CropShield withholds treatment advice and refers the user to their local Krishi Vigyan Kendra (KVK)."),
        ("Q9: What is the significance of Skip Connections in ResNet?",
         "In standard deep networks, gradients shrink exponentially as they backpropagate through many layers (Vanishing Gradient Problem). ResNet skip connections (F(x) + x) allow gradients to flow directly through identity shortcuts, enabling deeper networks to train efficiently without degradation."),
        ("Q10: What are the main limitations and future scopes of your project?",
         "Limitations: Real-world field accuracy (61.3% Top-1) is still lower than lab accuracy due to lighting and complex backgrounds, and we currently have only one insect pest class (Spider Mites). Future Work: Expanding the field dataset for Indian crops, deploying an offline mobile app via PyTorch Mobile/ONNX, and integrating automated disease severity percentage scoring.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", q_style))
        story.append(Paragraph(a, a_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {PDF_OUT}")

if __name__ == "__main__":
    build_pdf()
