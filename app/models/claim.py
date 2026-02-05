"""
Claim model - the core entity of the insurance claims system.
Represents an insurance claim from submission through settlement.
"""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, Enum as SQLEnum, JSON, Index, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
from datetime import datetime

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin, SoftDeleteMixin, AuditMixin


class ClaimType(str, enum.Enum):
    """Types of insurance claims we handle"""
    AUTO = "auto"  # Vehicle accidents and damage
    PROPERTY = "property"  # Home/building damage
    LIABILITY = "liability"  # Third-party claims


class ClaimStatus(str, enum.Enum):
    """
    Claim lifecycle status.
    Maps to our LangGraph workflow states.
    """
    SUBMITTED = "submitted"  # Just filed, not yet processed
    PROCESSING = "processing"  # In the workflow
    UNDER_REVIEW = "under_review"  # Needs detailed examination
    INVESTIGATING = "investigating"  # Potential fraud detected
    APPROVED = "approved"  # Ready for settlement
    DENIED = "denied"  # Claim rejected
    SETTLED = "settled"  # Payment issued
    CLOSED = "closed"  # Finalized


class ClaimPriority(str, enum.Enum):
    """Processing priority level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Claim(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """
    Insurance claim from initial submission to final settlement.
    Central entity that connects all other components.
    """
    __tablename__ = "claims"

    # Claim identification
    claim_number = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="Human-readable claim number (e.g., CLM-2024-001234)"
    )

    # Classification
    claim_type = Column(
        SQLEnum(ClaimType),
        nullable=False,
        index=True
    )

    status = Column(
        SQLEnum(ClaimStatus),
        default=ClaimStatus.SUBMITTED,
        nullable=False,
        index=True
    )

    priority = Column(
        SQLEnum(ClaimPriority),
        default=ClaimPriority.MEDIUM,
        nullable=False,
        index=True
    )

    # Relationships (claimant is optional for historical data)
    claimant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    policy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Policy this claim is filed under"
    )

    assigned_adjuster_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Adjuster assigned to review this claim"
    )

    # Incident details
    incident_date = Column(
        DateTime,
        nullable=False,
        index=True,
        comment="When the incident occurred"
    )

    incident_location = Column(
        String(500),
        nullable=True,
        comment="Where the incident happened"
    )

    description = Column(
        Text,
        nullable=False,
        comment="Claimant's description of what happened"
    )

    # Financial
    estimated_damage = Column(
        Float,
        nullable=False,
        comment="Claimant's initial damage estimate"
    )

    approved_amount = Column(
        Float,
        nullable=True,
        comment="Final approved settlement amount"
    )

    deductible = Column(
        Float,
        nullable=True,
        comment="Deductible amount from policy"
    )

    # AI/ML scores
    fraud_score = Column(
        Float,
        nullable=True,
        index=True,
        comment="Fraud probability (0.0 to 1.0) from ML model"
    )

    confidence_score = Column(
        Float,
        nullable=True,
        comment="System's confidence in automated decision"
    )

    # Workflow tracking
    workflow_state = Column(
        JSON,
        nullable=True,
        comment="LangGraph workflow state (serialized)"
    )

    auto_approved = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether claim was auto-approved by system"
    )

    # Dates
    submitted_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    reviewed_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    settled_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    # Additional metadata
    claim_metadata = Column(
        JSON,
        nullable=True,
        comment="Additional flexible data storage"
    )

    # Relationships
    claimant = relationship(
        "User",
        back_populates="claims",
        foreign_keys=[claimant_id]
    )

    assigned_adjuster = relationship(
        "User",
        back_populates="assigned_claims",
        foreign_keys=[assigned_adjuster_id]
    )

    policy = relationship("Policy", back_populates="claims")

    damage_assessments = relationship(
        "DamageAssessment",
        back_populates="claim",
        cascade="all, delete-orphan"
    )

    fraud_scores = relationship(
        "FraudScore",
        back_populates="claim",
        cascade="all, delete-orphan"
    )

    settlement = relationship(
        "Settlement",
        back_populates="claim",
        uselist=False,
        cascade="all, delete-orphan"
    )

    workflow_states = relationship(
        "WorkflowState",
        back_populates="claim",
        cascade="all, delete-orphan"
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index('ix_claims_status_submitted', 'status', 'submitted_at'),
        Index('ix_claims_claimant_status', 'claimant_id', 'status'),
        Index('ix_claims_adjuster_status', 'assigned_adjuster_id', 'status'),
        Index('ix_claims_fraud_score_status', 'fraud_score', 'status'),
    )

    def __repr__(self) -> str:
        return f"<Claim {self.claim_number} - {self.status.value}>"

    @property
    def is_high_value(self) -> bool:
        """Claims over $25,000 are considered high value"""
        return self.estimated_damage > 25000

    @property
    def is_suspicious(self) -> bool:
        """Check if fraud score indicates suspicion"""
        return self.fraud_score is not None and self.fraud_score > 0.5

    @property
    def processing_time_days(self) -> int:
        """How many days this claim has been open"""
        if self.closed_at:
            delta = self.closed_at - self.submitted_at
        else:
            delta = datetime.utcnow() - self.submitted_at
        return delta.days
