"""
SQLAlchemy ORM Models for ClaimGuard
"""

from app.models.base import TimestampMixin, UUIDMixin, SoftDeleteMixin, AuditMixin
from app.models.user import User, Role, RoleType
from app.models.policy import Policy
from app.models.claim import Claim, ClaimType, ClaimStatus, ClaimPriority
from app.models.damage_assessment import DamageAssessment, DamageType, DamageSeverity
from app.models.fraud_score import FraudScore
from app.models.settlement import Settlement
from app.models.workflow_state import WorkflowState

__all__ = [
    # Base mixins
    "TimestampMixin",
    "UUIDMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    # User models
    "User",
    "Role",
    "RoleType",
    # Policy
    "Policy",
    # Claim models
    "Claim",
    "ClaimType",
    "ClaimStatus",
    "ClaimPriority",
    # Damage assessment
    "DamageAssessment",
    "DamageType",
    "DamageSeverity",
    # Fraud detection
    "FraudScore",
    # Settlement
    "Settlement",
    # Workflow
    "WorkflowState",
]
