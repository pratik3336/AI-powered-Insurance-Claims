"""
ClaimGuard Gradio Dashboard
Interactive UI for insurance claims processing and fraud detection
"""

import gradio as gr
import pandas as pd
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from app.models.claim import Claim
from app.models.fraud_score import FraudScore
from app.models.policy import Policy
from app.services.ai.vision_analyzer import get_vision_analyzer


# Database connection
engine = create_engine(settings.DATABASE_URL)


def get_claims_overview():
    """Get overview statistics"""
    session = Session(engine)

    total_claims = session.query(Claim).count()
    high_risk = session.query(FraudScore).filter(FraudScore.fraud_score >= 0.5).count()
    pending_review = session.query(Claim).filter(Claim.status == 'PROCESSING').count()

    total_value = session.query(Claim.estimated_damage).count()

    session.close()

    return f"""
    ### 📊 ClaimGuard Dashboard

    **Total Claims:** {total_claims:,}

    **High-Risk Claims:** {high_risk:,} ({high_risk/total_claims*100:.1f}%)

    **Pending Review:** {pending_review:,}

    **Total Estimated Damages:** ${total_value:,.2f}
    """


def load_claims_data(risk_filter="all", limit=100):
    """Load claims data with filters"""

    session = Session(engine)

    query = session.query(
        Claim.claim_number,
        Claim.claim_type,
        Claim.status,
        Claim.estimated_damage,
        FraudScore.fraud_score,
        FraudScore.risk_level,
        Policy.policy_number
    ).join(FraudScore).join(Policy)

    if risk_filter == "high_risk":
        query = query.filter(FraudScore.fraud_score >= 0.5)
    elif risk_filter == "medium_risk":
        query = query.filter(FraudScore.fraud_score >= 0.3, FraudScore.fraud_score < 0.5)
    elif risk_filter == "low_risk":
        query = query.filter(FraudScore.fraud_score < 0.3)

    query = query.order_by(desc(FraudScore.fraud_score)).limit(limit)

    df = pd.read_sql(query.statement, session.bind)

    session.close()

    # Format columns
    if len(df) > 0:
        df['fraud_score'] = df['fraud_score'].apply(lambda x: f"{x:.3f}")
        df['estimated_damage'] = df['estimated_damage'].apply(lambda x: f"${x:,.2f}")
        df.columns = ['Claim #', 'Type', 'Status', 'Estimated Damage', 'Fraud Score', 'Risk Level', 'Policy #']

    return df


def get_claim_details(claim_number):
    """Get detailed information about a claim"""

    if not claim_number:
        return "Enter a claim number to view details"

    session = Session(engine)

    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()

    if not claim:
        session.close()
        return f"❌ Claim {claim_number} not found"

    fraud_score = session.query(FraudScore).filter(FraudScore.claim_id == claim.id).first()

    details = f"""
    ## Claim Details: {claim.claim_number}

    **Status:** {claim.status.value}
    **Priority:** {claim.priority.value}
    **Type:** {claim.claim_type.value}

    ### Incident Information
    **Date:** {claim.incident_date}
    **Location:** {claim.incident_location}
    **Description:** {claim.description}

    ### Financial
    **Estimated Damage:** ${claim.estimated_damage:,.2f}
    **Deductible:** ${claim.deductible:,.2f}

    ### Fraud Assessment
    **Fraud Score:** {fraud_score.fraud_score:.3f}
    **Risk Level:** {fraud_score.risk_level}
    **Requires Investigation:** {"Yes" if fraud_score.requires_investigation else "No"}

    **Fraud Flags:** {', '.join(fraud_score.fraud_flags) if fraud_score.fraud_flags else 'None'}

    ### Model Information
    **ML Score:** {fraud_score.ml_model_score:.3f}
    **Graph Score:** {fraud_score.graph_network_score:.3f}
    **Pattern Score:** {fraud_score.pattern_matching_score:.3f}
    """

    session.close()

    return details


def analyze_damage_yolo_subprocess(image_path):
    """Run YOLO inference via subprocess in clean venv"""
    import subprocess
    import json
    import tempfile

    venv_python = Path("/Users/aakashbhatt/ClaimGuard/venv_yolo/bin/python")

    if not venv_python.exists():
        return {
            "success": False,
            "error": "YOLO environment not set up. Run: bash scripts/setup_yolo_venv.sh"
        }

    # Create temp script for inference
    script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/aakashbhatt/ClaimGuard')

from ultralytics import YOLO
import json

model = YOLO('/Users/aakashbhatt/ClaimGuard/app/ml/models/yolo_damage/damage_detector_v1/weights/best.pt')
results = model('{image_path}')

detections = []
for result in results:
    for box in result.boxes:
        detections.append({{
            "class_name": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        }})

print(json.dumps({{"detections": detections}}))
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        result = subprocess.run(
            [str(venv_python), temp_script],
            capture_output=True,
            text=True,
            timeout=30
        )

        Path(temp_script).unlink()

        if result.returncode == 0:
            data = json.loads(result.stdout.strip().split('\n')[-1])
            return {"success": True, "detections": data['detections']}
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_damage_image(image, model_choice="OpenAI Vision"):
    """Analyze uploaded damage image"""

    if image is None:
        return "Please upload an image to analyze"

    try:
        if model_choice == "Custom YOLO":
            # Save image to temp file for YOLO
            from PIL import Image
            import tempfile

            if isinstance(image, str):
                image_path = image
            else:
                # Save numpy array to temp file
                pil_image = Image.fromarray(image)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    pil_image.save(f, format='JPEG')
                    image_path = f.name

            # Run YOLO inference
            result = analyze_damage_yolo_subprocess(image_path)

            if not isinstance(image, str):
                Path(image_path).unlink()

            if result.get('success'):
                detections = result['detections']

                if detections:
                    damage_types = list(set(d['class_name'] for d in detections))
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)

                    # Estimate severity and cost
                    if len(detections) >= 3:
                        severity = "Major"
                        cost_min, cost_max = 8000, 20000
                    elif len(detections) >= 2:
                        severity = "Moderate"
                        cost_min, cost_max = 2000, 8000
                    else:
                        severity = "Minor"
                        cost_min, cost_max = 500, 2000

                    output = f"""
### ✅ Damage Assessment Complete (Custom YOLO)

**Damage Types:** {', '.join(damage_types)}

**Detections:** {len(detections)} damage area(s) found

**Average Confidence:** {avg_conf:.1%}

**Severity:** {severity}

**Estimated Cost:** ${cost_min:,.0f} - ${cost_max:,.0f}

#### Detected Damages:
"""
                    for i, det in enumerate(detections, 1):
                        output += f"\n{i}. **{det['class_name']}** - Confidence: {det['confidence']:.1%}"

                    output += "\n\n---\n*Model: Custom YOLOv8 Nano (6 damage classes)*\n*Cost: $0 (Free local inference)*"
                    return output
                else:
                    return """
### ℹ️ No Damage Detected

The custom YOLO model did not detect any damage in this image.

Possible reasons:
- No visible damage present
- Damage type not in training data (6 classes)
- Confidence threshold not met

---
*Model: Custom YOLOv8 Nano*
"""
            else:
                return f"""
### ❌ YOLO Analysis Failed

{result.get('error', 'Unknown error')}

**Troubleshooting:**
1. Run: `bash scripts/setup_yolo_venv.sh`
2. Or use OpenAI Vision instead

---
*Falling back to OpenAI Vision is recommended*
"""

        else:  # OpenAI Vision
            analyzer = get_vision_analyzer()

            # Convert gradio image to bytes
            from PIL import Image
            import io

            if isinstance(image, str):
                # File path
                result = analyzer.analyze_damage(image)
            else:
                # numpy array from gradio
                pil_image = Image.fromarray(image)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                result = analyzer.analyze_damage(img_bytes)

            if result.get('success'):
                output = f"""
### ✅ Damage Assessment Complete (OpenAI Vision)

**Damage Types:** {', '.join(result.get('damage_types', []))}

**Severity:** {result.get('severity')}

**Severity Score:** {result.get('severity_score')}/100

**Affected Areas:** {', '.join(result.get('affected_areas', []))}

**Estimated Cost:** ${result.get('cost_estimate', {}).get('min', 0):,.2f} - ${result.get('cost_estimate', {}).get('max', 0):,.2f}

**Notes:** {result.get('notes', 'N/A')}

---
*Model: {result.get('model')}*

*Tokens Used: {result.get('usage', {}).get('total_tokens', 0)}*

*Cost: ~$0.0006*
"""
                return output
            else:
                return f"❌ Analysis failed: {result.get('error')}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="ClaimGuard - AI Insurance Claims Processing", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🛡️ ClaimGuard
    ## AI-Powered Insurance Claims Processing & Fraud Detection
    """)

    with gr.Tab("📊 Dashboard"):
        overview = gr.Markdown(get_claims_overview())
        gr.Button("🔄 Refresh").click(get_claims_overview, outputs=overview)

        gr.Markdown("### Recent Claims")

        risk_filter = gr.Radio(
            choices=["all", "high_risk", "medium_risk", "low_risk"],
            value="all",
            label="Filter by Risk Level"
        )

        claims_table = gr.Dataframe(
            value=load_claims_data(),
            label="Claims",
            interactive=False
        )

        risk_filter.change(load_claims_data, inputs=risk_filter, outputs=claims_table)

    with gr.Tab("🔍 Claim Details"):
        gr.Markdown("### Search Claim")

        claim_input = gr.Textbox(
            label="Enter Claim Number",
            placeholder="e.g., CLM-1994-000001"
        )

        claim_details_output = gr.Markdown()

        gr.Button("🔍 Search").click(
            get_claim_details,
            inputs=claim_input,
            outputs=claim_details_output
        )

    with gr.Tab("📸 Damage Assessment"):
        gr.Markdown("""
        ### Vehicle Damage Assessment
        Upload an image of vehicle damage for AI analysis
        """)

        with gr.Row():
            with gr.Column():
                model_selector = gr.Radio(
                    choices=["OpenAI Vision", "Custom YOLO"],
                    value="Custom YOLO",
                    label="Select AI Model",
                    info="Choose between OpenAI Vision API or Custom YOLO model"
                )

                damage_image = gr.Image(
                    label="Upload Damage Photo",
                    type="numpy"
                )

                analyze_btn = gr.Button("🔍 Analyze Damage", variant="primary")

            with gr.Column():
                damage_results = gr.Markdown(label="Assessment Results")

        analyze_btn.click(
            analyze_damage_image,
            inputs=[damage_image, model_selector],
            outputs=damage_results
        )

        gr.Markdown("""
        ### Model Comparison

        **Custom YOLO** (Recommended):
        - ✅ FREE (local inference)
        - ✅ Fast (~50ms per image)
        - ✅ Trained on 6 damage types
        - ✅ No API costs

        **OpenAI Vision**:
        - 💰 $0.0006 per image
        - 🔍 Detailed analysis
        - 🌍 General purpose
        - ⏱️ ~2-3 seconds per image
        """)

    with gr.Tab("📈 Analytics"):
        gr.Markdown("### Fraud Detection Analytics")

        session = Session(engine)

        # Risk distribution
        risk_dist = pd.read_sql("""
            SELECT
                risk_level,
                COUNT(*) as count,
                AVG(fraud_score) as avg_score
            FROM fraud_scores
            GROUP BY risk_level
            ORDER BY
                CASE risk_level
                    WHEN 'low' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'high' THEN 3
                    WHEN 'critical' THEN 4
                END
        """, session.bind)

        session.close()

        gr.Dataframe(
            value=risk_dist,
            label="Risk Level Distribution"
        )

    gr.Markdown("""
    ---
    **ClaimGuard v1.0** | Powered by Custom YOLO, OpenAI Vision & XGBoost
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛡️  Starting ClaimGuard Dashboard")
    print("="*60)
    print(f"\n📊 Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'local'}")
    print(f"🤖 AI Model: {settings.OPENAI_VISION_MODEL}")
    print("\n" + "="*60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
