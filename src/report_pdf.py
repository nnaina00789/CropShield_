import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image

def generate_pdf_report(user_img: Image.Image, heatmap_img: Image.Image, info: dict, conf: float, top3_list: list, kvk_info: dict = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1b5e20'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2e7d32'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#212121')
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#757575')
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🌿 CropShield — Crop Disease Advisory Report", title_style))
    date_str = datetime.now().strftime("%d %B %Y, %I:%M %p")
    story.append(Paragraph(f"<b>Generated on:</b> {date_str} &nbsp;|&nbsp; <b>Model:</b> ResNet18 Field-Adapted + LeafDetector", subtitle_style))
    story.append(Spacer(1, 4))

    # 2. Images Table (Original Photo + Grad-CAM Heatmap)
    # Save images to in-memory buffers for ReportLab
    img_buf = io.BytesIO()
    user_img.resize((180, 180)).save(img_buf, format="JPEG")
    img_buf.seek(0)
    rl_user = RLImage(img_buf, width=150, height=150)

    heat_buf = io.BytesIO()
    heatmap_img.resize((180, 180)).save(heat_buf, format="JPEG")
    heat_buf.seek(0)
    rl_heat = RLImage(heat_buf, width=150, height=150)

    # Diagnosis Summary Text Box
    diag_text = (
        f"<b>Detected Disease:</b> <font color='#1b5e20' size='11'><b>{info.get('name', 'N/A')}</b></font><br/><br/>"
        f"<b>Confidence:</b> {conf:.1%}<br/>"
        f"<b>Pathogen Type:</b> {info.get('type', 'N/A').title()}<br/>"
        f"<b>Leaf Validation:</b> Verified by LeafDetector (99.05% binary accuracy)<br/><br/>"
        f"<b>Top Predictions:</b><br/>"
    )
    for rank, (name, prob) in enumerate(top3_list[:3], 1):
        diag_text += f"&nbsp;&nbsp;{rank}. {name} — {prob:.1%}<br/>"

    img_table = Table(
        [
            [Paragraph("<b>Submitted Leaf Photo</b>", body_style), Paragraph("<b>Grad-CAM Attention Heatmap</b>", body_style), Paragraph("<b>Diagnosis Summary</b>", body_style)],
            [rl_user, rl_heat, Paragraph(diag_text, body_style)]
        ],
        colWidths=[160, 160, 220]
    )
    img_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
        ('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#f9fbe7')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#c8e6c9')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 3. Pathogen Cause & Symptoms
    story.append(Paragraph("1. Cause & Biological Origin", section_heading))
    story.append(Paragraph(info.get('cause', 'N/A'), body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Visible Symptoms", section_heading))
    story.append(Paragraph(info.get('symptoms', 'N/A'), body_style))
    story.append(Spacer(1, 4))

    # 4. Immediate Treatment Actions
    story.append(Paragraph("3. Recommended Management & Chemical/Organic Actions", section_heading))
    for step in info.get('treatment', []):
        story.append(Paragraph(f"• {step}", bullet_style))
    story.append(Spacer(1, 4))

    # 5. Long-term Prevention
    story.append(Paragraph("4. Prevention & Cultural Control Practices", section_heading))
    for step in info.get('prevention', []):
        story.append(Paragraph(f"• {step}", bullet_style))
    story.append(Spacer(1, 8))

    # 6. Local KVK Advisory Contact
    if kvk_info:
        story.append(Paragraph("5. Local Agricultural Support (Krishi Vigyan Kendra)", section_heading))
        kvk_text = (
            f"<b>Centre:</b> {kvk_info.get('kvk', 'N/A')}<br/>"
            f"<b>Covered Districts:</b> {kvk_info.get('district', 'N/A')}<br/>"
            f"<b>Contact:</b> {kvk_info.get('contact', 'N/A')}<br/>"
            f"<b>Helpline:</b> <font color='#b71c1c'><b>{kvk_info.get('helpline', '1800-180-1551')}</b></font>"
        )
        kvk_table = Table([[Paragraph(kvk_text, body_style)]], colWidths=[540])
        kvk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f8e9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#81c784')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(kvk_table)
        story.append(Spacer(1, 8))

    # 7. Disclaimer
    story.append(Paragraph(
        "<b>Important Advisory Disclaimer:</b> CropShield is an AI-assisted decision-support system. "
        "Chemical recommendations indicate functional groups only, never exact dosages. Always verify diagnosis with a "
        "local agricultural officer or your nearest Krishi Vigyan Kendra (KVK) before chemical application, and strictly adhere to product label safety instructions.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
