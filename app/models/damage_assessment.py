"""
Damage assessment model - stores AI analysis results from images/videos.
"""

from sqlalchemy import Column, String, Float, ForeignKey, JSON, Text, Enum as SQLEnum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class DamageType(str, enum.Enum):
    """Types of vehicle/property damage detected by AI"""
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    BROKEN = "broken"
    SHATTERED = "shattered"
    CRUSHED = "crushed"
    BURNED = "burned"
    WATER_DAMAGE = "water_damage"
    UNKNOWN = "unknown"


class DamageSeverity(str, enum.Enum):
    """AI-determined severity levels"""
    MINOR = "minor"  # Cosmetic, easy fix
    MODERATE = "moderate"  # Repairable
    MAJOR = "major"  # Extensive damage
    TOTAL_LOSS = "total_loss"  # Not repairable


class DamageAssessment(Base, UUIDMixin, TimestampMixin):
    """
    AI-generated damage assessment from images/videos.
    One claim can have multiple assessments (multiple photos).
    """
    __tablename__ = "damage_assessments"

    claim_id = Column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Source file
    file_url = Column(
        String(500),
        nullable=False,
        comment="URL to image/video in storage"
    )

    file_type = Column(
        String(20),
        nullable=False,
        comment="image or video"
    )

    # AI analysis results
    damage_type = Column(
        SQLEnum(DamageType),
        nullable=True,
        index=True
    )

    severity = Column(
        SQLEnum(DamageSeverity),
        nullable=True,
        index=True
    )

    severity_score = Column(
        Float,
        nullable=True,
        comment="0-100 scale, where 100 is total loss"
    )

    # Affected areas (for vehicles: front bumper, door, etc.)
    affected_areas = Column(
        JSON,
        nullable=True,
        comment="List of damaged parts/areas"
    )

    # Estimated repair cost from AI
    estimated_cost_min = Column(Float, nullable=True)
    estimated_cost_max = Column(Float, nullable=True)

    # Full AI response (for debugging and improvement)
    ai_response = Column(
        JSON,
        nullable=True,
        comment="Complete response from OpenAI Vision API"
    )

    # AI model used
    model_version = Column(
        String(50),
        nullable=True,
        comment="e.g., gpt-4o-mini"
    )

    confidence_score = Column(
        Float,
        nullable=True,
        comment="AI's confidence in assessment (0-1)"
    )

    # Human review
    reviewed = Column(Boolean, default=False, nullable=False)
    reviewer_notes = Column(Text, nullable=True)
    human_override = Column(
        Boolean,
        default=False,
        comment="True if adjuster changed AI assessment"
    )

    # Relationships
    claim = relationship("Claim", back_populates="damage_assessments")

    def __repr__(self) -> str:
        return f"<DamageAssessment {self.damage_type} - {self.severity}>"

    @property
    def estimated_cost_average(self) -> float:
        """Average of min/max cost estimates"""
        if self.estimated_cost_min and self.estimated_cost_max:
            return (self.estimated_cost_min + self.estimated_cost_max) / 2
        return 0.0

    @property
    def is_high_severity(self) -> bool:
        """Check if damage is major or total loss"""
        return self.severity in [DamageSeverity.MAJOR, DamageSeverity.TOTAL_LOSS]
