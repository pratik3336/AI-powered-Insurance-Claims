"""
Fraud score model - stores ML and graph-based fraud detection results.
"""

from sqlalchemy import Column, String, Float, ForeignKey, JSON, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class FraudScore(Base, UUIDMixin, TimestampMixin):
    """
    Fraud analysis results combining ML model and graph network analysis.
    One claim can have multiple scores as models are retrained.
    """
    __tablename__ = "fraud_scores"

    claim_id = Column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Overall fraud score (composite)
    fraud_score = Column(
        Float,
        nullable=False,
        index=True,
        comment="Composite fraud probability (0.0 to 1.0)"
    )

    # Individual component scores
    ml_model_score = Column(
        Float,
        nullable=True,
        comment="XGBoost/RandomForest prediction"
    )

    graph_network_score = Column(
        Float,
        nullable=True,
        comment="Neo4j graph analysis score"
    )

    pattern_matching_score = Column(
        Float,
        nullable=True,
        comment="Rule-based pattern detection"
    )

    # Fraud indicators/flags
    fraud_flags = Column(
        JSON,
        nullable=True,
        comment="List of suspicious patterns detected"
    )

    # Examples:
    # ["frequent_claims", "shared_vehicle", "suspicious_shop", "geographic_anomaly"]

    # Risk classification
    risk_level = Column(
        String(20),
        nullable=False,
        index=True,
        comment="low, medium, high, critical"
    )

    # Model metadata
    model_version = Column(
        String(50),
        nullable=True,
        comment="Version of fraud detection model used"
    )

    feature_importance = Column(
        JSON,
        nullable=True,
        comment="Which features contributed most to score"
    )

    # Neo4j graph findings
    fraud_network_id = Column(
        String(100),
        nullable=True,
        comment="ID of fraud ring/network in Neo4j"
    )

    network_connections = Column(
        JSON,
        nullable=True,
        comment="Related claimants, vehicles, shops"
    )

    # Investigation status
    requires_investigation = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )

    investigated = Column(Boolean, default=False, nullable=False)
    investigation_notes = Column(Text, nullable=True)

    # Final determination
    confirmed_fraud = Column(
        Boolean,
        nullable=True,
        comment="Human confirmation of fraud"
    )

    # Relationships
    claim = relationship("Claim", back_populates="fraud_scores")

    def __repr__(self) -> str:
        return f"<FraudScore {self.fraud_score:.2f} - {self.risk_level}>"

    @property
    def is_suspicious(self) -> bool:
        """Quick check if score indicates fraud risk"""
        return self.fraud_score >= 0.5

    @property
    def recommended_action(self) -> str:
        """Suggest next steps based on score"""
        if self.fraud_score >= 0.8:
            return "immediate_investigation"
        elif self.fraud_score >= 0.5:
            return "detailed_review"
        elif self.fraud_score >= 0.3:
            return "flag_for_monitoring"
        else:
            return "normal_processing"
