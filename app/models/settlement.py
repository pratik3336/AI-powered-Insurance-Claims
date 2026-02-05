"""
Settlement model - stores approved payment details and generated letters.
"""

from sqlalchemy import Column, String, Float, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class Settlement(Base, UUIDMixin, TimestampMixin):
    """
    Approved settlement details for a claim.
    One-to-one relationship with Claim.
    """
    __tablename__ = "settlements"

    claim_id = Column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )

    # Settlement amounts
    approved_amount = Column(
        Float,
        nullable=False,
        comment="Final approved settlement"
    )

    deductible_applied = Column(Float, nullable=False)
    net_payment = Column(
        Float,
        nullable=False,
        comment="Amount actually paid to claimant"
    )

    # AI-generated settlement letter
    settlement_letter = Column(
        Text,
        nullable=True,
        comment="AI-generated explanation letter"
    )

    letter_url = Column(
        String(500),
        nullable=True,
        comment="URL to PDF letter in storage"
    )

    # Payment tracking
    payment_method = Column(String(50), nullable=True)
    payment_reference = Column(String(100), nullable=True)
    paid_at = Column(DateTime, nullable=True)

    # Approval
    approved_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )

    approved_at = Column(DateTime, nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="settlement")

    def __repr__(self) -> str:
        return f"<Settlement ${self.net_payment:.2f}>"
