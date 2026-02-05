"""
Damage assessment service - integrates Vision AI with database
"""

from typing import List, Optional, Union
from pathlib import Path
from uuid import UUID
from sqlalchemy.orm import Session

from app.services.ai.vision_analyzer import get_vision_analyzer
from app.models.damage_assessment import DamageAssessment, DamageType, DamageSeverity
from app.models.claim import Claim


class DamageAssessmentService:
    """
    Service for analyzing vehicle damage and saving to database
    """

    def __init__(self, db: Session):
        self.db = db
        self.vision_analyzer = get_vision_analyzer()

    def assess_claim_damage(
        self,
        claim_id: UUID,
        image_sources: List[Union[str, Path, bytes]],
        file_urls: Optional[List[str]] = None
    ) -> List[DamageAssessment]:
        """
        Analyze damage images for a claim and save assessments

        Args:
            claim_id: Claim ID
            image_sources: List of image paths or bytes
            file_urls: Optional list of URLs where images are stored

        Returns:
            List of DamageAssessment objects
        """

        # Get claim for context
        claim = self.db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        # Build context for vision analysis
        claim_context = {
            'incident_description': claim.description,
            'claimed_damage': claim.estimated_damage
        }

        # Add vehicle info from metadata if available
        if claim.claim_metadata:
            if 'make' in claim.claim_metadata:
                claim_context['vehicle_make'] = claim.claim_metadata['make']

        assessments = []

        for idx, image_source in enumerate(image_sources):
            # Analyze image
            analysis = self.vision_analyzer.analyze_damage(
                image_source,
                claim_context
            )

            if not analysis.get('success'):
                # Log error but continue
                print(f"Failed to analyze image {idx}: {analysis.get('error')}")
                continue

            # Map severity string to enum
            severity_map = {
                'MINOR': DamageSeverity.MINOR,
                'MODERATE': DamageSeverity.MODERATE,
                'MAJOR': DamageSeverity.MAJOR,
                'TOTAL_LOSS': DamageSeverity.TOTAL_LOSS
            }
            severity = severity_map.get(
                analysis.get('severity'),
                DamageSeverity.MODERATE
            )

            # Map first damage type to enum (or use DENT as default)
            damage_types = analysis.get('damage_types', [])
            damage_type_map = {
                'scratch': DamageType.SCRATCH,
                'dent': DamageType.DENT,
                'crack': DamageType.CRACK,
                'broken': DamageType.BROKEN,
                'shattered': DamageType.SHATTERED,
                'crushed': DamageType.CRUSHED,
                'burned': DamageType.BURNED,
                'water damage': DamageType.WATER_DAMAGE,
                'water_damage': DamageType.WATER_DAMAGE
            }

            primary_damage_type = DamageType.DENT  # default
            for dt in damage_types:
                if dt.lower() in damage_type_map:
                    primary_damage_type = damage_type_map[dt.lower()]
                    break

            # Create assessment record
            assessment = DamageAssessment(
                claim_id=claim_id,
                file_url=file_urls[idx] if file_urls and idx < len(file_urls) else None,
                file_type='image',
                damage_type=primary_damage_type,
                severity=severity,
                severity_score=analysis.get('severity_score', 50),
                affected_areas=analysis.get('affected_areas', []),
                estimated_cost_min=analysis.get('cost_estimate', {}).get('min', 0),
                estimated_cost_max=analysis.get('cost_estimate', {}).get('max', 0),
                ai_response=analysis.get('raw_response', ''),
                model_version=analysis.get('model', 'gpt-4o-mini'),
                confidence_score=0.85,  # High confidence for vision model
                reviewed=False
            )

            self.db.add(assessment)
            assessments.append(assessment)

        # Commit all assessments
        self.db.commit()

        # Update claim with average damage estimate
        if assessments:
            avg_min = sum(a.estimated_cost_min for a in assessments) / len(assessments)
            avg_max = sum(a.estimated_cost_max for a in assessments) / len(assessments)
            avg_estimate = (avg_min + avg_max) / 2

            # Update claim if AI estimate differs significantly
            if claim.estimated_damage:
                # If AI estimate is very different, flag for review
                diff_pct = abs(avg_estimate - claim.estimated_damage) / claim.estimated_damage
                if diff_pct > 0.5:  # More than 50% difference
                    print(f"⚠️  Large discrepancy: Claimed ${claim.estimated_damage:,.2f} vs AI ${avg_estimate:,.2f}")

        return assessments

    def get_claim_assessments(self, claim_id: UUID) -> List[DamageAssessment]:
        """Get all damage assessments for a claim"""
        return self.db.query(DamageAssessment).filter(
            DamageAssessment.claim_id == claim_id
        ).all()

    def get_assessment_summary(self, claim_id: UUID) -> dict:
        """Get summarized damage assessment for a claim"""
        assessments = self.get_claim_assessments(claim_id)

        if not assessments:
            return {
                'assessment_count': 0,
                'max_severity': None,
                'total_cost_estimate': {'min': 0, 'max': 0},
                'all_damage_types': [],
                'all_affected_areas': []
            }

        # Aggregate severity
        severity_order = {
            DamageSeverity.MINOR: 1,
            DamageSeverity.MODERATE: 2,
            DamageSeverity.MAJOR: 3,
            DamageSeverity.TOTAL_LOSS: 4
        }
        max_severity = max(assessments, key=lambda a: severity_order.get(a.severity, 0)).severity

        # Aggregate costs
        total_min = sum(a.estimated_cost_min for a in assessments)
        total_max = sum(a.estimated_cost_max for a in assessments)

        # Collect all damage types and areas
        all_damage_types = set()
        all_affected_areas = set()
        for assessment in assessments:
            all_damage_types.add(assessment.damage_type.value)
            if assessment.affected_areas:
                all_affected_areas.update(assessment.affected_areas)

        return {
            'assessment_count': len(assessments),
            'max_severity': max_severity.value,
            'total_cost_estimate': {
                'min': total_min,
                'max': total_max,
                'avg': (total_min + total_max) / 2
            },
            'all_damage_types': list(all_damage_types),
            'all_affected_areas': list(all_affected_areas),
            'reviewed': all(a.reviewed for a in assessments)
        }
