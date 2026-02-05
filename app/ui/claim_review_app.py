"""
ClaimGuard Claim Review Dashboard

A unified workflow for insurance claim adjusters to review and process claims efficiently.

Key Features:
1. Click-to-select claims from queue
2. Real-time fraud risk calculation
3. Detailed damage analysis with cost breakdowns
4. Professional approval letter generation
5. Formal rejection email with fraud indicators
"""

import gradio as gr
import pandas as pd
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session
import sys
from pathlib import Path
from typing import Tuple, List
from uuid import UUID
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.config import settings
from app.models.claim import Claim, ClaimStatus
from app.models.fraud_score import FraudScore
from app.models.policy import Policy
from app.models.damage_assessment import DamageAssessment
from app.services.ai.vision_analyzer import get_vision_analyzer

# Database connection
engine = create_engine(settings.DATABASE_URL)


def get_pending_claims(status_filter="pending_review", limit=50):
    """Get claims needing review"""
    session = Session(engine)

    query = session.query(
        Claim.claim_number,
        Claim.claim_type,
        Claim.status,
        Claim.estimated_damage,
        Claim.incident_date,
        FraudScore.fraud_score,
        FraudScore.risk_level
    ).join(FraudScore)

    if status_filter == "pending_review":
        query = query.filter(Claim.status.in_([ClaimStatus.PROCESSING, ClaimStatus.UNDER_REVIEW]))
    elif status_filter == "high_risk":
        query = query.filter(FraudScore.fraud_score >= 0.5)
    elif status_filter == "has_images":
        query = query.join(DamageAssessment).filter(DamageAssessment.file_url != None)

    query = query.order_by(desc(FraudScore.fraud_score)).limit(limit)

    df = pd.read_sql(query.statement, session.bind)
    session.close()

    if len(df) > 0:
        df['fraud_score'] = df['fraud_score'].apply(lambda x: f"{x*100:.1f}%")
        df['estimated_damage'] = df['estimated_damage'].apply(lambda x: f"${x:,.2f}")
        df['incident_date'] = pd.to_datetime(df['incident_date']).dt.strftime('%Y-%m-%d')
        df.columns = ['Claim #', 'Type', 'Status', 'Damage', 'Date', 'Fraud Score', 'Risk']

    return df


def select_claim_from_table(evt: gr.SelectData, dataframe):
    """Handle claim selection from table click"""
    if evt.index is not None and len(dataframe) > 0:
        row_index = evt.index[0]
        claim_number = dataframe.iloc[row_index]['Claim #']
        return claim_number
    return ""


def load_claim(claim_number: str) -> Tuple[str, str, List]:
    """Load claim details"""
    if not claim_number:
        return "", "❌ Select a claim from the table or enter a claim number", []

    session = Session(engine)
    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()

    if not claim:
        session.close()
        return "", f"❌ Claim {claim_number} not found", []

    fraud_score = session.query(FraudScore).filter(FraudScore.claim_id == claim.id).first()
    policy = session.query(Policy).filter(Policy.id == claim.policy_id).first()

    # Get damage images
    damage_assessments = session.query(DamageAssessment).filter(
        DamageAssessment.claim_id == claim.id
    ).all()

    image_paths = []
    for assessment in damage_assessments:
        if assessment.file_url and Path(assessment.file_url).exists():
            image_paths.append(str(assessment.file_url))

    details = f"""
# 📋 Step 1: Claim Information

## Claim: {claim.claim_number}

### Basic Information
- **Status**: {claim.status.value.upper()}
- **Type**: {claim.claim_type.value.title()}
- **Priority**: {claim.priority.value.title()}
- **Claimed Amount**: ${claim.estimated_damage:,.2f}
- **Deductible**: ${claim.deductible:,.2f}

### Incident Details
- **Date**: {claim.incident_date}
- **Location**: {claim.incident_location}
- **Description**: {claim.description}

### Policyholder Information
- **Name**: {policy.policyholder_name if policy else 'N/A'}
- **Policy Number**: {policy.policy_number if policy else 'N/A'}
- **Coverage Limit**: ${(policy.coverage_limit if policy else 0):,.2f}

### Evidence
- **Damage Photos**: {len(image_paths)} image(s) attached

---

✅ **Claim loaded successfully** - Proceed to Step 2 to analyze damage
"""

    session.close()
    return str(claim.id), details, image_paths


def analyze_damage_detailed(claim_id_str: str, model_choice: str) -> str:
    """Step 2: Detailed damage analysis with per-part breakdown"""

    if not claim_id_str:
        return "❌ No claim loaded. Complete Step 1 first."

    try:
        claim_id = UUID(claim_id_str)
    except:
        return "❌ Invalid claim ID"

    session = Session(engine)
    assessments = session.query(DamageAssessment).filter(
        DamageAssessment.claim_id == claim_id
    ).all()

    if not assessments:
        session.close()
        return "❌ No damage images found for this claim"

    if model_choice == "Custom YOLO":
        session.close()
        return """
# 📸 Step 2: Damage Analysis

⚠️ **Custom YOLO requires additional setup**

Please use **OpenAI Vision (RAG)** for complete analysis.
"""

    # OpenAI Vision (RAG) Analysis
    analyzer = get_vision_analyzer()

    output = f"""
# 📸 Step 2: Detailed Damage Analysis

## Analysis Method
- **Model**: OpenAI Vision (gpt-4o-mini) + RAG
- **Images Analyzed**: {len(assessments)}
- **Data Source**: Historical Repair Database

## Damage Breakdown by Part

"""

    total_min = 0
    total_max = 0
    all_parts = []

    for i, assessment in enumerate(assessments, 1):
        if assessment.file_url and Path(assessment.file_url).exists():
            result = analyzer.analyze_damage(assessment.file_url)

            if result.get('success'):
                damage_types = result.get('damage_types', [])
                affected_areas = result.get('affected_areas', [])
                cost_est = result.get('cost_estimate', {})
                severity = result.get('severity', 'UNKNOWN')

                output += f"""
### Image {i}: {Path(assessment.file_url).name}

**Severity**: {severity}
**Damage Types**: {', '.join(damage_types) if damage_types else 'None detected'}

**Affected Parts**:
"""

                # Per-part breakdown
                for area in affected_areas:
                    # Estimate per-part cost (simplified distribution)
                    part_min = cost_est.get('min', 0) / max(len(affected_areas), 1)
                    part_max = cost_est.get('max', 0) / max(len(affected_areas), 1)
                    output += f"- **{area.title()}**: ${part_min:,.0f} - ${part_max:,.0f}\n"
                    all_parts.append(area)

                total_min += cost_est.get('min', 0)
                total_max += cost_est.get('max', 0)

                output += f"\n**Image Subtotal**: ${cost_est.get('min', 0):,.2f} - ${cost_est.get('max', 0):,.2f}\n\n"

    output += f"""
---

## Total Repair Cost Estimate

**All Parts**: {', '.join(set(all_parts)) if all_parts else 'None detected'}

**Estimated Total Cost**: ${total_min:,.2f} - ${total_max:,.2f}

**Cost Breakdown**:
- Parts & Materials: ~70% (${total_min*0.7:,.2f} - ${total_max*0.7:,.2f})
- Labor: ~25% (${total_min*0.25:,.2f} - ${total_max*0.25:,.2f})
- Other: ~5% (${total_min*0.05:,.2f} - ${total_max*0.05:,.2f})

---

✅ **Damage analysis complete** - Proceed to Step 3 to calculate fraud risk
"""

    session.close()
    return output


def calculate_fraud_realtime(claim_id_str: str) -> Tuple[str, str]:
    """Step 3: Calculate fraud score in real-time

    Returns: (fraud_analysis_md, auto_denial_reason)
    """

    if not claim_id_str:
        return "❌ No claim loaded. Complete Step 1 first.", ""

    try:
        claim_id = UUID(claim_id_str)
    except:
        return "❌ Invalid claim ID", ""

    session = Session(engine)
    claim = session.query(Claim).filter(Claim.id == claim_id).first()
    fraud_score = session.query(FraudScore).filter(FraudScore.claim_id == claim_id).first()

    if not claim or not fraud_score:
        session.close()
        return "❌ Claim data not found", ""

    # Get actual damage estimate from assessments (if analyzed)
    assessments = session.query(DamageAssessment).filter(
        DamageAssessment.claim_id == claim_id
    ).all()

    actual_damage_min = sum(a.estimated_cost_min or 0 for a in assessments)
    actual_damage_max = sum(a.estimated_cost_max or 0 for a in assessments)

    output = f"""
# 🚨 Step 3: Intelligent Fraud Risk Assessment

## Real-Time Analysis

### 1. Claim Amount Verification
- **Claimed Amount**: ${claim.estimated_damage:,.2f}
- **Actual Damage Estimate**: ${actual_damage_min:,.2f} - ${actual_damage_max:,.2f}
"""

    # Check for claim amount discrepancies
    discrepancy_flags = []
    if actual_damage_min > 0:  # Only if damage was analyzed
        if claim.estimated_damage < actual_damage_min * 0.5:
            discrepancy_flags.append(f"⚠️ **Under-claiming**: Claimed ${claim.estimated_damage:,.2f} but damage is ${actual_damage_min:,.2f}-${actual_damage_max:,.2f}")
            output += f"- ⚠️ **SUSPICIOUS**: Claim is significantly **LOWER** than actual damage\n"
        elif claim.estimated_damage > actual_damage_max * 1.5:
            discrepancy_flags.append(f"🚨 **Over-claiming**: Claimed ${claim.estimated_damage:,.2f} but damage only ${actual_damage_min:,.2f}-${actual_damage_max:,.2f}")
            output += f"- 🚨 **FRAUD INDICATOR**: Claim is **150%+ HIGHER** than actual damage\n"
        else:
            output += f"- ✅ **Reasonable**: Claim amount matches damage estimate\n"

    # Evidence quality check
    num_images = len(assessments)
    output += f"\n### 2. Evidence Quality\n"
    output += f"- **Photos Submitted**: {num_images} image(s)\n"

    if num_images == 0:
        discrepancy_flags.append("🚨 **No Evidence**: No damage photos submitted")
        output += "- 🚨 **CRITICAL**: No damage evidence provided\n"
    elif num_images < 2:
        output += "- ⚠️ **Limited Evidence**: Few photos submitted\n"
    else:
        output += "- ✅ **Adequate Evidence**: Sufficient documentation\n"

    # Policy coverage check
    policy = session.query(Policy).filter(Policy.id == claim.policy_id).first()
    output += f"\n### 3. Policy Coverage Analysis\n"
    if policy:
        claim_ratio = claim.estimated_damage / policy.coverage_limit
        output += f"- **Coverage Limit**: ${policy.coverage_limit:,.2f}\n"
        output += f"- **Claim %**: {claim_ratio*100:.1f}% of limit\n"

        if claim_ratio > 0.95:
            discrepancy_flags.append(f"⚠️ **Maxing Coverage**: Claim is {claim_ratio*100:.1f}% of policy limit")
            output += f"- ⚠️ **SUSPICIOUS**: Claiming near maximum coverage\n"
        else:
            output += f"- ✅ **Normal**: Within reasonable range\n"

    # Historical pattern check (from pre-calculated fraud score)
    output += f"\n### 4. Historical Patterns (ML Analysis)\n"
    output += f"- **ML Risk Score**: {fraud_score.ml_model_score*100:.1f}%\n"
    output += f"- **Pattern Matching**: {fraud_score.pattern_matching_score*100:.1f}%\n"

    historical_flags = fraud_score.fraud_flags or []
    if historical_flags:
        output += f"\n**Historical Red Flags**:\n"
        for flag in historical_flags:
            output += f"- {flag}\n"

    # Calculate overall risk
    all_flags = discrepancy_flags + list(historical_flags)

    # Smart risk calculation
    if discrepancy_flags:
        # Real fraud indicators found
        final_score = min(fraud_score.fraud_score + 0.15, 1.0)
        risk_level = "CRITICAL" if len(discrepancy_flags) >= 2 else "HIGH"
        risk_color = "🔴" if risk_level == "CRITICAL" else "🟠"
    else:
        # No real issues, use base score
        final_score = fraud_score.fraud_score
        if final_score >= 0.7:
            risk_level = "HIGH"
            risk_color = "🟠"
        elif final_score >= 0.3:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"

    # Generate recommendation
    if len(discrepancy_flags) >= 2 or final_score >= 0.8:
        recommendation = "**DENY CLAIM**"
        action_color = "🚨"
    elif len(discrepancy_flags) >= 1 or final_score >= 0.5:
        recommendation = "**MANUAL REVIEW REQUIRED**"
        action_color = "⚠️"
    else:
        recommendation = "**APPROVE CLAIM**"
        action_color = "✅"

    output += f"""

---

## {action_color} Final Assessment

**Overall Risk**: {risk_color} **{risk_level}** ({final_score*100:.1f}%)

**All Issues Detected** ({len(all_flags)}):
"""

    if all_flags:
        for flag in all_flags:
            output += f"- {flag}\n"
    else:
        output += "✅ No fraud indicators detected\n"

    output += f"""

---

## {action_color} Recommendation: {recommendation}

"""

    # Generate auto-denial reason
    auto_denial_reason = ""
    if len(discrepancy_flags) >= 1 or final_score >= 0.7:
        auto_denial_reason = f"""Based on comprehensive fraud analysis, this claim exhibits multiple red flags:

FRAUD RISK ASSESSMENT:
- Overall fraud score: {final_score*100:.1f}% ({risk_level} risk)
- {len(all_flags)} fraud indicator(s) detected

SPECIFIC ISSUES IDENTIFIED:
"""
        for i, flag in enumerate(all_flags[:5], 1):
            auto_denial_reason += f"{i}. {flag}\n"

        auto_denial_reason += f"""
CONCLUSION:
This claim shows signs of potential fraud and does not meet our approval criteria. The discrepancies identified warrant denial pending further investigation.

Policyholder has the right to appeal this decision with additional documentation within 30 days."""

    output += "\n✅ **Assessment complete** - Proceed to Step 4 to make decision"

    session.close()
    return output, auto_denial_reason


def generate_approval_letter(claim_id_str: str) -> str:
    """Step 4a: Generate approval letter"""

    if not claim_id_str:
        return "❌ No claim loaded"

    try:
        claim_id = UUID(claim_id_str)
    except:
        return "❌ Invalid claim ID"

    session = Session(engine)
    claim = session.query(Claim).filter(Claim.id == claim_id).first()
    policy = session.query(Policy).filter(Policy.id == claim.policy_id).first()

    if not claim:
        session.close()
        return "❌ Claim not found"

    # Update status
    claim.status = ClaimStatus.APPROVED
    session.commit()

    settlement = claim.estimated_damage - claim.deductible
    approval_date = datetime.now().strftime("%B %d, %Y")

    letter = f"""
# ✅ CLAIM APPROVED

---

## OFFICIAL APPROVAL LETTER FOR FUND DISBURSEMENT

**ClaimGuard Insurance Company**
123 Insurance Plaza, Suite 500
Claims Department
Phone: (800) 555-CLAIM | Fax: (800) 555-0199
Email: claims@claimguard.com

---

**Date**: {approval_date}

**Claim Number**: {claim.claim_number}
**Policy Number**: {policy.policy_number if policy else 'N/A'}
**Policyholder**: {policy.policyholder_name if policy else 'N/A'}

---

### RE: APPROVAL OF INSURANCE CLAIM

Dear {policy.policyholder_name if policy else 'Policyholder'},

We are pleased to inform you that your insurance claim has been **APPROVED** following a thorough review of your submitted documentation and damage assessment.

### CLAIM DETAILS

**Incident Date**: {claim.incident_date}
**Incident Location**: {claim.incident_location}
**Claim Type**: {claim.claim_type.value.upper()}

**Incident Description**:
{claim.description}

### SETTLEMENT BREAKDOWN

| Item | Amount |
|------|--------|
| **Total Estimated Damage** | ${claim.estimated_damage:,.2f} |
| **Less: Deductible** | -${claim.deductible:,.2f} |
| **APPROVED SETTLEMENT AMOUNT** | **${settlement:,.2f}** |

### PAYMENT INFORMATION

Your settlement payment of **${settlement:,.2f}** will be processed within 5-7 business days.

**Payment Method**: Direct deposit or check (as per your preference on file)

**Payment Schedule**:
- Payment Authorization: {approval_date}
- Expected Payment Date: {(datetime.now() + pd.Timedelta(days=7)).strftime("%B %d, %Y")}

### NEXT STEPS

1. ✅ **Completed** - Claim review and approval
2. ✅ **Completed** - Settlement calculation
3. ⏳ **In Progress** - Payment processing
4. ⏳ **Pending** - Fund disbursement (5-7 business days)

### IMPORTANT NOTES

- Please retain this letter for your records
- Settlement is based on actual repair costs as assessed
- Any additional damage discovered during repairs may require supplemental claim
- Original receipts should be kept for tax purposes

### QUESTIONS?

If you have any questions about this approval or settlement, please contact our Claims Department:

- **Phone**: (800) 555-CLAIM
- **Email**: claims@claimguard.com
- **Claim Number Reference**: {claim.claim_number}

---

Thank you for your patience during the claims process. We're glad we could help resolve this matter promptly.

**Sincerely**,

**Claims Review Department**
ClaimGuard Insurance Company

---

*This is an official approval document. Please keep for your records.*

**Approval Code**: {claim.claim_number}-APP-{datetime.now().strftime("%Y%m%d")}
"""

    session.close()
    return letter


def generate_rejection_email(claim_id_str: str, denial_reason: str) -> str:
    """Step 4b: Generate rejection email"""

    if not claim_id_str:
        return "❌ No claim loaded"

    if not denial_reason or len(denial_reason.strip()) < 10:
        return "❌ Please provide a detailed denial reason (minimum 10 characters)"

    try:
        claim_id = UUID(claim_id_str)
    except:
        return "❌ Invalid claim ID"

    session = Session(engine)
    claim = session.query(Claim).filter(Claim.id == claim_id).first()
    policy = session.query(Policy).filter(Policy.id == claim.policy_id).first()
    fraud_score = session.query(FraudScore).filter(FraudScore.claim_id == claim_id).first()

    if not claim:
        session.close()
        return "❌ Claim not found"

    # Update status
    claim.status = ClaimStatus.DENIED
    session.commit()

    denial_date = datetime.now().strftime("%B %d, %Y")

    email = f"""
# ❌ CLAIM DENIED

---

## CLAIM DENIAL NOTIFICATION EMAIL

**FROM**: ClaimGuard Claims Department <claims@claimguard.com>
**TO**: {policy.policyholder_name if policy else 'Policyholder'} <policyholder@email.com>
**SUBJECT**: Claim Denial Notice - Claim #{claim.claim_number}
**DATE**: {denial_date}

---

Dear {policy.policyholder_name if policy else 'Policyholder'},

We regret to inform you that after careful review, your insurance claim has been **DENIED**.

### CLAIM INFORMATION

**Claim Number**: {claim.claim_number}
**Policy Number**: {policy.policy_number if policy else 'N/A'}
**Incident Date**: {claim.incident_date}
**Claim Amount**: ${claim.estimated_damage:,.2f}
**Denial Date**: {denial_date}

### REASON FOR DENIAL

{denial_reason}

### REVIEW SUMMARY

Our comprehensive review included:
- ✓ Damage evidence analysis
- ✓ Fraud risk assessment (Score: {fraud_score.fraud_score*100:.1f}%)
- ✓ Policy coverage verification
- ✓ Claims history review

**Risk Assessment**: {fraud_score.risk_level.upper()}

**Fraud Indicators Detected**:
"""

    if fraud_score.fraud_flags:
        for flag in fraud_score.fraud_flags:
            email += f"- {flag}\n"
    else:
        email += "- None detected\n"

    email += f"""

### YOUR RIGHTS

You have the right to:

1. **Appeal this Decision**
   - Submit additional documentation
   - Request independent review
   - File formal appeal within 30 days

2. **Request Explanation**
   - Schedule call with claims adjuster
   - Receive detailed denial breakdown
   - Review specific policy clauses

3. **File Complaint**
   - State Insurance Commissioner
   - Better Business Bureau
   - Internal Escalation Department

### HOW TO APPEAL

If you believe this denial is in error, you may file an appeal:

**Appeal Deadline**: {(datetime.now() + pd.Timedelta(days=30)).strftime("%B %d, %Y")}

**Appeal Process**:
1. Submit written appeal to: appeals@claimguard.com
2. Include claim number: {claim.claim_number}
3. Provide additional evidence or documentation
4. Response within 15 business days

**Appeal Contact**:
- **Email**: appeals@claimguard.com
- **Phone**: (800) 555-APPEAL
- **Mail**: ClaimGuard Appeals Dept, PO Box 9999, Claims City, ST 12345

### NEED ASSISTANCE?

**Claims Department**:
- Phone: (800) 555-CLAIM
- Email: claims@claimguard.com
- Hours: Monday-Friday, 8AM-6PM EST

**Reference Number**: {claim.claim_number}

---

We understand this is disappointing news. Our decision is based on thorough investigation and policy terms. If you have questions or wish to discuss this denial, please don't hesitate to contact us.

**Sincerely**,

**Claims Review Department**
ClaimGuard Insurance Company

---

*This is an official claim denial notification. Please retain for your records.*

**Denial Code**: {claim.claim_number}-DENY-{datetime.now().strftime("%Y%m%d")}
**Case Closed**: {denial_date}
"""

    session.close()
    return email


# Create Gradio interface
with gr.Blocks(title="ClaimGuard - Enhanced Claim Review") as app:

    gr.Markdown("""
    # 🛡️ ClaimGuard - Enhanced Claim Review System
    ### Professional Claims Adjuster Workflow with Document Generation
    """)

    # Hidden state
    current_claim_id = gr.State("")

    with gr.Row():
        # LEFT: Claims Queue
        with gr.Column(scale=1):
            gr.Markdown("## 📋 Claims Queue")
            gr.Markdown("*Click on a row to select a claim*")

            status_filter = gr.Radio(
                choices=["pending_review", "high_risk", "has_images", "all"],
                value="pending_review",
                label="Filter"
            )

            claims_table = gr.Dataframe(
                value=get_pending_claims("pending_review"),
                label="Claims - Click to Select",
                interactive=False
            )

            refresh_btn = gr.Button("🔄 Refresh", size="sm")

        # RIGHT: Workflow
        with gr.Column(scale=2):

            # STEP 1: Load Claim
            with gr.Accordion("📋 Step 1: Select & Load Claim", open=True):
                gr.Markdown("**👈 Click any row in the Claims Queue to automatically load**")
                claim_number_input = gr.Textbox(
                    label="Selected Claim Number",
                    placeholder="Click a row in the table on the left →",
                    interactive=False
                )
                claim_details = gr.Markdown("👈 **Click a claim from the table to begin**")
                claim_images = gr.Gallery(
                    label="Damage Evidence Photos",
                    columns=3,
                    type="filepath"
                )

            # STEP 2: Analyze Damage
            with gr.Accordion("📸 Step 2: Analyze Damage (Detailed)", open=False):
                model_selector = gr.Radio(
                    choices=["OpenAI Vision (RAG)", "Custom YOLO"],
                    value="OpenAI Vision (RAG)",
                    label="Analysis Model"
                )
                analyze_btn = gr.Button("🔍 Analyze Damage", variant="primary", size="lg")
                damage_output = gr.Markdown()

            # STEP 3: Calculate Fraud
            with gr.Accordion("🚨 Step 3: Calculate Fraud Risk (Real-Time)", open=False):
                fraud_btn = gr.Button("🔍 Calculate Fraud Score", variant="primary", size="lg")
                fraud_output = gr.Markdown()

            # STEP 4: Decision
            with gr.Accordion("⚖️ Step 4: Make Decision & Generate Documents", open=False):
                gr.Markdown("### Choose an action:")
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve & Generate Letter", variant="primary", size="lg")
                    deny_btn = gr.Button("❌ Deny & Generate Email", variant="stop", size="lg")

                denial_reason = gr.Textbox(
                    label="Denial Reason (required if denying)",
                    placeholder="Provide detailed explanation for claim denial...",
                    lines=4,
                    visible=False
                )

                deny_submit_btn = gr.Button("📧 Generate Rejection Email", visible=False, variant="stop")

                decision_output = gr.Markdown()

    gr.Markdown("""
    ---
    **ClaimGuard v3.0** | Enhanced Workflow | Real-Time Analysis | Document Generation
    """)

    # Event handlers
    status_filter.change(get_pending_claims, inputs=status_filter, outputs=claims_table)
    refresh_btn.click(get_pending_claims, inputs=status_filter, outputs=claims_table)

    # Click to select claim from table - auto-loads claim
    claims_table.select(
        select_claim_from_table,
        inputs=claims_table,
        outputs=claim_number_input
    ).then(
        load_claim,
        inputs=claim_number_input,
        outputs=[current_claim_id, claim_details, claim_images]
    )

    analyze_btn.click(
        analyze_damage_detailed,
        inputs=[current_claim_id, model_selector],
        outputs=damage_output
    )

    fraud_btn.click(
        calculate_fraud_realtime,
        inputs=current_claim_id,
        outputs=[fraud_output, denial_reason]
    )

    approve_btn.click(
        generate_approval_letter,
        inputs=current_claim_id,
        outputs=decision_output
    )

    # Show denial reason box when deny button clicked
    deny_btn.click(
        lambda: [gr.Textbox(visible=True), gr.Button(visible=True)],
        outputs=[denial_reason, deny_submit_btn]
    )

    # Generate rejection email when reason submitted
    deny_submit_btn.click(
        generate_rejection_email,
        inputs=[current_claim_id, denial_reason],
        outputs=decision_output
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛡️  ClaimGuard - Enhanced Claim Review Dashboard V3")
    print("="*60)
    print("\n✨ New Features:")
    print("  ✓ Click to select claims from table")
    print("  ✓ Real-time fraud score calculation")
    print("  ✓ Detailed per-part damage analysis")
    print("  ✓ Professional approval letter generation")
    print("  ✓ Formal rejection email generation")
    print("\n" + "="*60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
