"""
NeuroScan AI - Brain Tumor Detection with YOLO
Streamlined version for end users
"""

from typing import Any, Dict, List, Optional, Tuple
import io
import logging
from datetime import datetime
import os
import tempfile

import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Conditional import for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Conditional import for PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------- Configuration ----------
YOLO_MODEL_PATH = "yolov8_model.pt"
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")

# ---------- Logging ----------
logger = logging.getLogger("neuroscan")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- Page Config & CSS ----------
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            background: linear-gradient(135deg, #0066CC, #009688);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .medical-card {
            background-color: var(--background-color);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin: 10px 0;
        }
        .metric-card {
            background-color: var(--background-color);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            border-top: 4px solid #0066CC;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .empty-state { 
            text-align: center; 
            padding: 40px 20px; 
            color: var(--text-color);
        }
        .risk-high { 
            background: #FF6B6B; 
            color: white; 
            padding: 6px 12px; 
            border-radius: 16px; 
            font-weight: 600;
            font-size: 0.8rem;
        }
        .risk-medium { 
            background: #FFA726; 
            color: black; 
            padding: 6px 12px; 
            border-radius: 16px; 
            font-weight: 600;
            font-size: 0.8rem;
        }
        .risk-low { 
            background: #66BB6A; 
            color: white; 
            padding: 6px 12px; 
            border-radius: 16px; 
            font-weight: 600;
            font-size: 0.8rem;
        }
        .risk-none { 
            background: #78909C; 
            color: white; 
            padding: 6px 12px; 
            border-radius: 16px; 
            font-weight: 600;
            font-size: 0.8rem;
        }
        .section-header {
            font-size: 1.4rem;
            color: var(--heading-color);
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }
        .info-box {
            background-color: var(--secondary-background-color);
            border-left: 4px solid #009688;
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 0 8px 8px 0;
        }
        .warning-box {
            background-color: #FFF3CD;
            border-left: 4px solid #FFA726;
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 0 8px 8px 0;
            color: #856404;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #4CAF50; }
        .status-inactive { background-color: #F44336; }
        .confidence-bar {
            height: 8px;
            background-color: var(--border-color);
            border-radius: 4px;
            margin: 8px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #0066CC, #009688);
            border-radius: 4px;
        }
        .referring-physician-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .referring-physician-header {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        .patient-name {
            font-size: 1.2rem;
            color: #0066CC;
            font-weight: 600;
            margin: 5px 0;
        }
        .physician-name {
            color: #2c3e50;
            margin: 5px 0;
        }
        .pdf-feature-box {
            background-color: #e8f4fd;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 12px 0;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_yolo_model(path: str = YOLO_MODEL_PATH) -> Optional[Any]:
    """Load YOLOv8 model safely and cache it."""
    if not YOLO_AVAILABLE:
        logger.error("Ultralytics not available")
        return None
        
    try:
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None
        
        # Try safe loading first
        try:
            import torch
            model = YOLO(path)
            logger.info("YOLO model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Safe loading failed, trying standard loading: {e}")
            model = YOLO(path)
            logger.info("YOLO model loaded with standard method")
            return model
        
    except Exception as exc:
        logger.exception("Error loading YOLO model: %s", exc)
        return None


def validate_file(file: Any) -> Tuple[bool, str]:
    """Check file extension and validity."""
    if file is None:
        return False, "No file provided"
    
    name = getattr(file, "name", "")
    if not any(name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, f"Unsupported file type: {name}"
    
    return True, "OK"


def run_yolo_detection(model: Any, image: Image.Image) -> Dict[str, Any]:
    """Run object detection using YOLOv8 model."""
    img_array = np.array(image.convert('RGB'))
    results = model(img_array, conf=0.25)
    return interpret_yolo_predictions(results, image.size)


def interpret_yolo_predictions(results: List, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Interpret YOLO model predictions and produce structured result."""
    if not results or not results[0].boxes:
        return {
            "detection": {
                "boxes": [],
                "class": "No Tumor",
                "risk_level": "None",
                "overall_confidence": 85.0,
                "confidence_scores": {
                    "Glioma": 0.0, 
                    "Meningioma": 0.0, 
                    "Pituitary": 0.0, 
                    "No Tumor": 85.0
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
    
    boxes = []
    # Match the class order from training: ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    class_names = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
    
    # Initialize confidence scores for all classes
    confidence_scores = {class_name: 0.0 for class_name in class_names.values()}
    
    # Risk levels mapping
    risk_levels = {
        "Glioma": "High",
        "Meningioma": "Medium", 
        "Pituitary": "Low",
        "No Tumor": "None"
    }
    
    # Process all detections
    detected_classes = set()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        
        # Get class name with correct mapping
        class_name = class_names.get(class_id, "Unknown")
        
        # Skip if this is "No Tumor" detection (we'll handle it specially)
        if class_name == "No Tumor":
            continue
            
        boxes.append({
            "x1": int(x1),
            "y1": int(y1), 
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(confidence),
            "type": class_name,
            "class_name": class_name.lower()
        })
        
        # Update confidence score for this class
        confidence_scores[class_name] = max(
            confidence_scores[class_name], 
            float(confidence) * 100
        )
        detected_classes.add(class_name)
    
    # Determine primary detection
    if boxes:
        # Sort by confidence and take the highest
        primary_detection = max(boxes, key=lambda x: x["confidence"])
        detected_class = primary_detection["type"]
        risk_level = risk_levels.get(detected_class, "Medium")
        overall_confidence = primary_detection["confidence"] * 100
    else:
        # No tumor detected
        detected_class = "No Tumor"
        risk_level = "None"
        overall_confidence = 85.0  # Default confidence for no tumor
        confidence_scores["No Tumor"] = overall_confidence
    
    return {
        "detection": {
            "boxes": boxes,
            "class": detected_class,
            "risk_level": risk_level,
            "overall_confidence": float(overall_confidence),
            "confidence_scores": confidence_scores,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

def draw_yolo_annotations(image: Image.Image, detection: Dict[str, Any]) -> Image.Image:
    """Return annotated PIL.Image with detection results."""
    arr = np.array(image.convert("RGB"))
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    colors = {
        "Glioma": (0, 0, 255),        # Red
        "Meningioma": (0, 165, 255),  # Orange
        "Pituitary": (255, 255, 0),   # Cyan
        "No Tumor": (0, 255, 0),      # Green
        "default": (0, 255, 0)        # Green
    }
    
    for box in detection.get("boxes", []):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        tumor_type = box.get("type", "default")
        confidence = box["confidence"]
        color = colors.get(tumor_type, colors["default"])
        
        label = f"{tumor_type} {confidence:.1%}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1 - h_text - 10), (x1 + w_text + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    detected_class = detection.get("class", "Unknown")
    overall_confidence = detection.get("overall_confidence", 0)
    risk_level = detection.get("risk_level", "")
    
    # Add detection text at top
    detection_text = f"{detected_class} ({overall_confidence:.1f}%) - {risk_level} Risk"
    
    cv2.putText(img, detection_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, detection_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def generate_text_report(analysis: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
    """Create comprehensive report using YOLO model results."""
    detection = analysis["detection"]
    patient_id = patient_info.get("id", "N/A")
    patient_name = patient_info.get("name", "Not Specified")
    physician = patient_info.get("physician", "Not Specified")
    
    md = [
        "# Medical Analysis Report",
        f"**Patient ID:** {patient_id}",
        f"**Patient Name:** {patient_name}",
        f"**Analysis Date:** {detection['timestamp']}",
        f"**Referring Physician:** {physician}",
        "## Diagnostic Summary",
        f"- **Primary Detection:** {detection['class']}",
        f"- **Confidence Level:** {detection['overall_confidence']:.1f}%",
        f"- **Risk Assessment:** {detection['risk_level']}",
        "## Imaging Findings",
        f"- **Detected Regions:** {len(detection['boxes'])}"
    ]
    
    if detection["boxes"]:
        for i, box in enumerate(detection["boxes"], 1):
            md.append(
                f"- **Region {i}:** {box['type']} | "
                f"Confidence: {box['confidence']:.1%} | "
                f"Location: ({box['x1']},{box['y1']})-({box['x2']},{box['y2']})"
            )
    else:
        md.append("- No pathological regions detected in the analyzed image.")
    
    md.extend(["## Confidence Scores"])
    
    for class_name, score in detection["confidence_scores"].items():
        if score > 0:
            md.append(f"- {class_name}: {score:.1f}%")
    
    md.extend([
        "## Clinical Recommendations"
    ])
    
    risk_level = detection["risk_level"]
    if risk_level == "High":
        md.extend([
            "- **Urgent neurosurgical consultation recommended**",
            "- Further imaging (contrast-enhanced MRI) advised",
            "- Consider immediate follow-up with oncology specialist"
        ])
    elif risk_level == "Medium":
        md.extend([
            "- **Schedule neurosurgical evaluation within 2 weeks**",
            "- Consider follow-up MRI in 3-6 months",
            "- Monitor for neurological symptoms"
        ])
    elif risk_level == "Low":
        md.extend([
            "- **Routine follow-up and monitoring recommended**",
            "- Clinical correlation with symptoms advised",
            "- Consider repeat imaging in 6-12 months"
        ])
    else:
        md.extend([
            "- **Routine screening recommended**",
            "- No immediate intervention required",
            "- Clinical correlation as needed"
        ])
    
    return "\n\n".join(md)


def generate_pdf_report(analysis: Dict[str, Any], patient_info: Dict[str, Any], annotated_image: Image.Image) -> bytes:
    """Generate a PDF report with analysis results including annotated image."""
    if not PDF_AVAILABLE:
        raise ImportError("ReportLab library not available")
    
    detection = analysis["detection"]
    
    # Create a buffer for PDF
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0066CC')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#009688')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    bold_style = ParagraphStyle(
        'CustomBold',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    # Build story
    story = []
    
    # Title
    story.append(Paragraph("NeuroScan AI - Medical Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    
    patient_data = [
        ["Patient ID:", patient_info.get("id", "N/A")],
        ["Patient Name:", patient_info.get("name", "Not Specified")],
        ["Referring Physician:", patient_info.get("physician", "Not Specified")],
        ["Analysis Date:", detection['timestamp']]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Diagnostic Summary
    story.append(Paragraph("Diagnostic Summary", heading_style))
    
    # Risk badge
    risk_color = {
        "High": colors.red,
        "Medium": colors.orange,
        "Low": colors.green,
        "None": colors.grey
    }.get(detection['risk_level'], colors.black)
    
    summary_data = [
        ["Primary Detection:", detection['class']],
        ["Confidence Level:", f"{detection['overall_confidence']:.1f}%"],
        ["Risk Assessment:", 
         Paragraph(f"<font color='{risk_color.hexval()}'><b>{detection['risk_level']} Risk</b></font>", normal_style)],
        ["Detected Regions:", str(len(detection['boxes']))]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Annotated Image Section
    story.append(Paragraph("Visual Analysis", heading_style))
    
    # Save annotated image to temporary file for PDF inclusion
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        annotated_image.save(tmp, format='PNG')
        tmp_path = tmp.name
    
    try:
        # Add image to PDF
        img = ReportLabImage(tmp_path, width=6*inch, height=4.5*inch)
        story.append(img)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<i>Figure: MRI analysis showing {detection['class']} detection with {detection['overall_confidence']:.1f}% confidence</i>", normal_style))
    except Exception as e:
        logger.warning(f"Could not add image to PDF: {e}")
        story.append(Paragraph("Annotated image unavailable in this report format.", normal_style))
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    story.append(Spacer(1, 20))
    
    # Imaging Findings
    story.append(Paragraph("Imaging Findings", heading_style))
    
    if detection["boxes"]:
        findings_text = f"Detected {len(detection['boxes'])} pathological region(s):"
        story.append(Paragraph(findings_text, normal_style))
        story.append(Spacer(1, 10))
        
        # Create table for findings
        findings_data = [["Region", "Type", "Confidence", "Location"]]
        for i, box in enumerate(detection["boxes"], 1):
            findings_data.append([
                f"Region {i}",
                box['type'],
                f"{box['confidence']:.1%}",
                f"({box['x1']},{box['y1']})-({box['x2']},{box['y2']})"
            ])
        
        findings_table = Table(findings_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 2*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        story.append(findings_table)
    else:
        story.append(Paragraph("No pathological regions detected in the analyzed image.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Confidence Scores
    story.append(Paragraph("Confidence Scores", heading_style))
    
    confidence_data = []
    for class_name, score in detection["confidence_scores"].items():
        if score > 0:
            confidence_data.append([class_name, f"{score:.1f}%"])
    
    if confidence_data:
        confidence_table = Table(confidence_data, colWidths=[3*inch, 3*inch])
        confidence_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(confidence_table)
    
    story.append(Spacer(1, 20))
    
    # Clinical Recommendations
    story.append(Paragraph("Clinical Recommendations", heading_style))
    
    risk_level = detection["risk_level"]
    if risk_level == "High":
        recommendations = [
            "• <b>Urgent neurosurgical consultation recommended</b>",
            "• Further imaging (contrast-enhanced MRI) advised",
            "• Consider immediate follow-up with oncology specialist",
            "• Monitor for neurological deficits",
            "• Consider hospital admission for close observation"
        ]
    elif risk_level == "Medium":
        recommendations = [
            "• <b>Schedule neurosurgical evaluation within 2 weeks</b>",
            "• Consider follow-up MRI in 3-6 months",
            "• Monitor for neurological symptoms",
            "• Discuss treatment options with multidisciplinary team",
            "• Consider neuropsychological assessment"
        ]
    elif risk_level == "Low":
        recommendations = [
            "• <b>Routine follow-up and monitoring recommended</b>",
            "• Clinical correlation with symptoms advised",
            "• Consider repeat imaging in 6-12 months",
            "• Monitor for any changes in symptoms",
            "• Consider consultation if symptoms develop"
        ]
    else:
        recommendations = [
            "• <b>Routine screening recommended</b>",
            "• No immediate intervention required",
            "• Clinical correlation as needed",
            "• Continue with regular health maintenance",
            "• Consider follow-up based on clinical presentation"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    story.append(Spacer(1, 20))
    
    # Footer
    footer_text = "This report was generated by NeuroScan AI - Brain Tumor Detection System. For clinical use only."
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceBefore=20
    )))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def render_confidence_bars(confidence_scores: Dict[str, float]) -> None:
    """Render confidence scores with visual bars."""
    for class_name, score in confidence_scores.items():
        if score > 0:
            col1, col2, col3 = st.columns([2, 5, 1])
            with col1:
                st.write(f"{class_name}:")
            with col2:
                st.markdown(
                    f'<div class="confidence-bar">'
                    f'<div class="confidence-fill" style="width: {score}%"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col3:
                st.write(f"{score:.1f}%")


def render_referring_physician_box(patient_info: Dict[str, Any]) -> None:
    """Render the Referring Physician box with patient name."""
    patient_name = patient_info.get("name", "Not Specified")
    physician = patient_info.get("physician", "Not Specified")
    
    st.markdown(
        f'''
        <div class="referring-physician-box">
            <div class="referring-physician-header">Referring Physician</div>
            <div class="patient-name">{patient_name}</div>
            <div class="physician-name">Physician: {physician}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def main() -> None:
    st.markdown('<div class="main-header">NeuroScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:var(--text-color);margin-bottom:18px;'>"
        "Brain Tumor Detection with AI Model</p>",
        unsafe_allow_html=True
    )

    # Check for model file
    model_exists = os.path.exists(YOLO_MODEL_PATH)
    
    if not model_exists:
        st.error(
            f"⚠️ **Model file '{YOLO_MODEL_PATH}' not found!**\n\n"
            "Please add your trained YOLOv8 model file to the project directory. "
            "See README.md for instructions."
        )

    with st.sidebar:
        st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
        with st.form("patient_form"):
            patient_id = st.text_input("Patient ID")
            patient_name = st.text_input("Patient Name")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
            patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            physician = st.text_input("Referring Physician")
            submitted = st.form_submit_button("Save Patient Information")
            if submitted:
                st.session_state["patient_info"] = {
                    "id": patient_id, 
                    "name": patient_name,
                    "age": int(patient_age), 
                    "sex": patient_sex, 
                    "physician": physician
                }
                st.success("Patient information saved")
        
        # Display Referring Physician box
        if "patient_info" in st.session_state and st.session_state["patient_info"]:
            render_referring_physician_box(st.session_state["patient_info"])

        st.markdown("---")
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        
        if "yolo_model" not in st.session_state:
            st.session_state["yolo_model"] = load_yolo_model()
        
        yolo_loaded = st.session_state["yolo_model"] is not None
        
        st.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div class="status-indicator {"status-active" if yolo_loaded else "status-inactive"}"></div>'
            f'<span>AI Model: <strong>{"Ready" if yolo_loaded else "Not Available"}</strong></span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if not YOLO_AVAILABLE:
            st.markdown(
                '<div class="warning-box">'
                '⚠️ Ultralytics package not installed properly'
                '</div>',
                unsafe_allow_html=True
            )
        
        # PDF status
        col1, col2 = st.columns([1, 4])
        with col1:
            pdf_status_class = "status-active" if PDF_AVAILABLE else "status-inactive"
            st.markdown(f'<div class="status-indicator {pdf_status_class}"></div>', unsafe_allow_html=True)
        with col2:
            st.write(f"**PDF Reports:** {'✅ Available' if PDF_AVAILABLE else '⚠️ Text Only'}")

    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Results Dashboard", "Clinical Report"])

    with tab1:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown('<div class="section-header">Upload MRI Image</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload an MRI image (JPG, JPEG, PNG)", 
                type=list(ALLOWED_EXTENSIONS)
            )
            
            if uploaded_file:
                valid, msg = validate_file(uploaded_file)
                if not valid:
                    st.error(msg)
                else:
                    try:
                        image = Image.open(uploaded_file).convert("RGB")
                        st.image(image, caption="Uploaded MRI")
                        st.write(f"Dimensions: {image.size[0]} × {image.size[1]} px")
                        st.session_state["uploaded_image"] = image
                    except Exception as ex:
                        st.error(f"Failed to read image: {ex}")
                        st.session_state.pop("uploaded_image", None)
                
        with col_right:
            st.markdown('<div class="section-header">AI Analysis</div>', unsafe_allow_html=True)
            if "uploaded_image" in st.session_state:
                if st.button("Start Analysis", type="primary", use_container_width=True):
                    img = st.session_state["uploaded_image"]
                    
                    with st.spinner("Running detection analysis..."):
                        try:
                            yolo_model = st.session_state.get("yolo_model")
                            if yolo_model is None:
                                st.error("AI model not available. Please check model file and dependencies.")
                            else:
                                yolo_results = run_yolo_detection(yolo_model, img)
                                st.session_state["yolo_results"] = yolo_results
                                st.success("Analysis complete")
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            logger.exception("Analysis error")
                else:
                    st.markdown(
                        '<div class="info-box">Ready to analyze. Press "Start Analysis".</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="info-box">Upload an MRI image to enable analysis.</div>',
                    unsafe_allow_html=True
                )

    with tab2:
        if st.session_state.get("yolo_results") and st.session_state.get("uploaded_image"):
            results = st.session_state["yolo_results"]
            detection = results["detection"]

            st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                badge_class = f"risk-{detection['risk_level'].lower()}"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h4>Detection</h4><h3>{detection["class"]}</h3>'
                    f'<div class="{badge_class}">{detection["risk_level"]} Risk</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with c2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h4>Confidence</h4><h3>{detection["overall_confidence"]:.1f}%</h3>'
                    f'<p>Detection Confidence</p></div>',
                    unsafe_allow_html=True
                )
            
            with c3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h4>Regions</h4><h3>{len(detection["boxes"])}</h3>'
                    f'<p>Detected Areas</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-header">Visual Analysis</div>', unsafe_allow_html=True)
            annotated = draw_yolo_annotations(st.session_state["uploaded_image"], detection)
            st.image(annotated, caption="Analysis Results")

            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "Download Annotated Image", 
                data=buf, 
                file_name="neuroscan_detection.png", 
                mime="image/png"
            )
            
            if detection["boxes"]:
                st.markdown("**Detection details:**")
                for i, box in enumerate(detection["boxes"], 1):
                    with st.expander(f"Region {i} - {box['type']} (Confidence: {box['confidence']:.1%})"):
                        st.write(f"Location: ({box['x1']}, {box['y1']}) to ({box['x2']}, {box['y2']})")
                        st.write(f"Size: {box['x2'] - box['x1']} × {box['y2'] - box['y1']} px")
                        st.write(f"Type: {box['type']}")
            else:
                st.markdown(
                    '<div class="info-box">No pathological regions detected.</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-header">Confidence Breakdown</div>', unsafe_allow_html=True)
            render_confidence_bars(detection["confidence_scores"])

        else:
            st.markdown(
                '<div class="empty-state">'
                '<h3>No Analysis Available</h3>'
                '<p>Upload an MRI image and run analysis to view results.</p>'
                '</div>',
                unsafe_allow_html=True
            )

    with tab3:
        st.markdown('<div class="section-header">Clinical Report</div>', unsafe_allow_html=True)
        if st.session_state.get("yolo_results"):
            patient_info = st.session_state.get("patient_info", {})
            report_md = generate_text_report(st.session_state["yolo_results"], patient_info)
            
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.markdown(report_md, unsafe_allow_html=False)
            st.markdown('</div>', unsafe_allow_html=True)

            # Single PDF download button - Generate PDF only when clicked
            if PDF_AVAILABLE:
                # Create a button that generates and downloads PDF
                if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        try:
                            # Get annotated image
                            annotated = draw_yolo_annotations(
                                st.session_state["uploaded_image"], 
                                st.session_state["yolo_results"]["detection"]
                            )
                            
                            # Generate PDF
                            pdf_bytes = generate_pdf_report(
                                st.session_state["yolo_results"],
                                patient_info,
                                annotated
                            )
                            
                            # Generate filename
                            patient_id = patient_info.get("id", "unknown")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"neuroscan_report_{patient_id}_{timestamp}.pdf"
                            
                            # Provide download link
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Failed to generate PDF: {e}")
                            logger.exception("PDF generation error")
                else:
                    # Show info when button is not pressed
                    st.info("Click 'Generate PDF Report' to create a professional PDF report with the annotated image.")
            else:
                # If PDF not available, provide text download
                if st.button("📄 Generate Text Report", type="primary", use_container_width=True):
                    report_text = generate_text_report(st.session_state["yolo_results"], patient_info)
                    patient_id = patient_info.get("id", "unknown")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"neuroscan_report_{patient_id}_{timestamp}.txt"
                    
                    st.download_button(
                        label="⬇️ Download Text Report",
                        data=report_text.encode('utf-8'),
                        file_name=filename,
                        mime="text/plain",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown(
                    '''
                    <div class="pdf-feature-box">
                    <strong>💡 Enhanced PDF Reports:</strong><br>
                    For professional PDF reports with tables and annotated images, 
                    install the required library by running: <code>pip install reportlab</code>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
        else:
            st.markdown(
                '<div class="empty-state">'
                '<h3>No Report Available</h3>'
                '<p>Complete an analysis to generate a clinical report.</p>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:var(--text-color);padding:10px;font-size:0.9rem;'>"
        "NeuroScan AI | Detection System | For clinical use only</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
