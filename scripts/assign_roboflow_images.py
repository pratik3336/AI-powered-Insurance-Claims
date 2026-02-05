"""
Assign Roboflow damage images to all claims
Each claim gets 1-2 random damage images from the dataset
"""

import sys
from pathlib import Path
import random
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.claim import Claim
from app.models.damage_assessment import DamageAssessment


def get_all_roboflow_images():
    """Get all image paths from Roboflow dataset"""
    base_path = Path("/Users/aakashbhatt/ClaimGuard/data/raw/roboflow_damage")

    image_paths = []

    # Get images from train, valid, and test sets
    for split in ['train', 'valid', 'test']:
        split_path = base_path / split / 'images'
        if split_path.exists():
            images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))
            image_paths.extend(images)

    return image_paths


def assign_images_to_claims():
    """Assign 1-2 random images to each claim"""

    engine = create_engine(settings.DATABASE_URL)
    session = Session(engine)

    # Get all image paths
    all_images = get_all_roboflow_images()
    print(f"📸 Found {len(all_images)} damage images from Roboflow dataset")

    if not all_images:
        print("❌ No images found! Check the path.")
        return

    # Get all claims
    claims = session.query(Claim).all()
    print(f"📋 Found {len(claims)} claims")

    # Delete existing damage assessments (clean slate)
    print("🗑️  Clearing existing damage assessments...")
    session.query(DamageAssessment).delete()
    session.commit()

    # Assign images to claims
    total_assigned = 0

    for i, claim in enumerate(claims, 1):
        # Randomly assign 1-2 images per claim
        num_images = random.choice([1, 1, 2])  # More likely to get 1 image

        # Select random images (with replacement since we have fewer images than claims)
        selected_images = random.sample(all_images, min(num_images, len(all_images)))

        for img_path in selected_images:
            assessment = DamageAssessment(
                claim_id=claim.id,
                file_type='image',
                file_url=str(img_path),
                reviewer_notes=f"Damage evidence from Roboflow dataset"
            )
            session.add(assessment)
            total_assigned += 1

        if i % 1000 == 0:
            print(f"  ✓ Processed {i:,} claims...")
            session.commit()

    # Final commit
    session.commit()

    print(f"\n✅ SUCCESS!")
    print(f"   📸 Assigned {total_assigned:,} images")
    print(f"   📋 Across {len(claims):,} claims")
    print(f"   📊 Average: {total_assigned/len(claims):.1f} images per claim")

    # Show sample
    print(f"\n📋 Sample claim with images:")
    sample_claim = claims[0]
    sample_assessments = session.query(DamageAssessment).filter(
        DamageAssessment.claim_id == sample_claim.id
    ).all()
    print(f"   Claim: {sample_claim.claim_number}")
    for assessment in sample_assessments:
        print(f"   - {Path(assessment.file_url).name}")

    session.close()


if __name__ == "__main__":
    print("="*60)
    print("🛡️  Assigning Roboflow Images to Claims")
    print("="*60)
    print()

    assign_images_to_claims()

    print()
    print("="*60)
