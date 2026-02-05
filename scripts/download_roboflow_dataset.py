"""
Download Roboflow car damage dataset and link to claims
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.models.claim import Claim
from app.models.fraud_score import FraudScore
from app.models.damage_assessment import DamageAssessment, DamageType, DamageSeverity


def download_roboflow_dataset():
    """Download car damage dataset from Roboflow"""

    print("\n📥 Downloading Roboflow Car Damage Dataset...")
    print("="*60)

    try:
        from roboflow import Roboflow

        # Initialize Roboflow (no API key needed for public datasets)
        rf = Roboflow(api_key="roboflow")  # Public access

        print("🔗 Accessing Roboflow workspace...")
        project = rf.workspace("car-damage-kadad").project("car-damage-images")

        print("📦 Downloading dataset version 3...")
        dataset = project.version(3).download("folder", location="data/raw/roboflow_damage")

        print(f"\n✅ Dataset downloaded to: data/raw/roboflow_damage")

        return dataset.location

    except Exception as e:
        print(f"\n❌ Error downloading from Roboflow: {e}")
        print("\n⚠️  Roboflow API method failed. Using alternative approach...")
        print("\nPlease download manually:")
        print("1. Visit: https://universe.roboflow.com/car-damage-kadad/car-damage-images/dataset/3")
        print("2. Click 'Download Dataset'")
        print("3. Select 'Folder Structure' format")
        print("4. Extract to: data/raw/roboflow_damage/")
        return None


def find_roboflow_images():
    """Find all images in the downloaded dataset"""

    dataset_dir = Path("data/raw/roboflow_damage")

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return []

    # Roboflow typically organizes as: train/, valid/, test/
    image_files = []

    for subdir in ['train', 'valid', 'test']:
        subdir_path = dataset_dir / subdir
        if subdir_path.exists():
            images = list(subdir_path.glob("*.jpg")) + \
                    list(subdir_path.glob("*.jpeg")) + \
                    list(subdir_path.glob("*.png"))
            image_files.extend(images)

    # Also check root directory
    root_images = list(dataset_dir.glob("*.jpg")) + \
                  list(dataset_dir.glob("*.jpeg")) + \
                  list(dataset_dir.glob("*.png"))
    image_files.extend(root_images)

    return image_files


def link_images_to_claims(image_files, max_images=100):
    """Link downloaded images to high-risk claims"""

    print(f"\n📎 Linking {min(len(image_files), max_images)} images to claims...")
    print("="*60)

    engine = create_engine(settings.DATABASE_URL)
    session = Session(engine)

    # Get existing assessments to avoid duplicates
    existing_claim_ids = set(
        c[0] for c in session.query(DamageAssessment.claim_id).distinct().all()
    )

    # Get high-risk claims without assessments
    high_risk_claims = session.query(Claim).join(FraudScore).filter(
        FraudScore.fraud_score >= 0.4,
        ~Claim.id.in_(existing_claim_ids)
    ).limit(max_images).all()

    if len(high_risk_claims) < max_images:
        print(f"⚠️  Only {len(high_risk_claims)} claims available without existing assessments")
        print("Adding more claims...")

        # Get additional claims
        additional = session.query(Claim).filter(
            ~Claim.id.in_(existing_claim_ids)
        ).limit(max_images - len(high_risk_claims)).all()

        high_risk_claims.extend(additional)

    linked = 0
    damage_types = [DamageType.DENT, DamageType.SCRATCH, DamageType.CRACK, DamageType.BROKEN]
    severities = [DamageSeverity.MINOR, DamageSeverity.MODERATE, DamageSeverity.MAJOR]

    for idx, (claim, img_path) in enumerate(zip(high_risk_claims, image_files[:max_images])):
        # Vary damage types and severities for realism
        damage_type = damage_types[idx % len(damage_types)]
        severity = severities[idx % len(severities)]

        severity_scores = {
            DamageSeverity.MINOR: (20, 40),
            DamageSeverity.MODERATE: (40, 70),
            DamageSeverity.MAJOR: (70, 95)
        }

        score_range = severity_scores[severity]
        severity_score = (score_range[0] + score_range[1]) // 2

        # Create assessment
        assessment = DamageAssessment(
            claim_id=claim.id,
            file_url=str(img_path.absolute()),
            file_type='image',
            damage_type=damage_type,
            severity=severity,
            severity_score=severity_score,
            affected_areas=['pending_ai_analysis'],
            estimated_cost_min=1000,
            estimated_cost_max=5000,
            ai_response='Roboflow dataset image - ready for AI analysis',
            model_version='roboflow_v3',
            confidence_score=0.0,
            reviewed=False
        )

        session.add(assessment)

        if (idx + 1) % 10 == 0:
            print(f"  Linked {idx + 1}/{min(len(image_files), max_images)} images...")

        linked += 1

    session.commit()
    session.close()

    print(f"\n✅ Successfully linked {linked} images to claims!")

    return linked


def main():
    print("\n🛡️  ClaimGuard - Roboflow Dataset Integration")
    print("="*60)

    # Download dataset
    dataset_location = download_roboflow_dataset()

    # Find images
    print("\n🔍 Scanning for images...")
    image_files = find_roboflow_images()

    if not image_files:
        print("\n❌ No images found in data/raw/roboflow_damage/")
        print("\nPlease:")
        print("1. Download dataset manually from:")
        print("   https://universe.roboflow.com/car-damage-kadad/car-damage-images/dataset/3")
        print("2. Extract to: data/raw/roboflow_damage/")
        print("3. Run this script again")
        return

    print(f"✅ Found {len(image_files)} images")

    # Link to claims
    linked = link_images_to_claims(image_files, max_images=100)

    print("\n" + "="*60)
    print("✅ Dataset Integration Complete!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"  - Total images found: {len(image_files)}")
    print(f"  - Images linked to claims: {linked}")
    print(f"  - Dataset location: data/raw/roboflow_damage/")

    print("\n📋 Next Steps:")
    print("1. Open dashboard: http://localhost:7860")
    print("2. Go to 'Claim Details' tab")
    print("3. Search claims to view linked images")
    print("4. Go to 'Damage Assessment' tab")
    print("5. Upload images for AI analysis")

    print("\n💡 Tip: You can now analyze any of these images using OpenAI Vision!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
