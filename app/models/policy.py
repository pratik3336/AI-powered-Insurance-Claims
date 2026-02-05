"""
Policy models for insurance coverage and policy documents.
Simplified for internal tool - focuses on coverage validation.
"""

from sqlalchemy import Column, String, Float, Date, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class Policy(Base, UUIDMixin, TimestampMixin):
    """
    Insurance policy information.
    Used for coverage validation and RAG system.
    """
    __tablename__ = "policies"

    policy_number = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    # Policyholder info (simple strings, not user relationships)
    policyholder_name = Column(String(255), nullable=False)
    policyholder_email = Column(String(255), nullable=True)

    # Policy details
    policy_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="auto, property, liability"
    )

    coverage_limit = Column(
        Float,
        nullable=False,
        comment="Maximum coverage amount"
    )

    deductible = Column(
        Float,
        nullable=False,
        comment="Deductible amount"
    )

    # Dates
    effective_date = Column(Date, nullable=False, index=True)
    expiration_date = Column(Date, nullable=False, index=True)

    # Coverage details (JSON for flexibility)
    coverage_details = Column(
        JSON,
        nullable=True,
        comment="Detailed coverage information for RAG"
    )

    # Policy document (for RAG system)
    document_url = Column(
        String(500),
        nullable=True,
        comment="URL to policy document in storage"
    )

    is_active = Column(Boolean, default=True, nullable=False, index=True)

    # Relationships
    claims = relationship("Claim", back_populates="policy")

    def __repr__(self) -> str:
        return f"<Policy {self.policy_number}>"

    @property
    def is_valid(self) -> bool:
        """Check if policy is currently valid"""
        from datetime import date
        today = date.today()
        return (
            self.is_active and
            self.effective_date <= today <= self.expiration_date
        )
