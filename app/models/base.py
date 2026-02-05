"""
Base models and mixins for common database fields.
These patterns are reused across all tables to maintain consistency.
"""

from datetime import datetime
from typing import Any
from sqlalchemy import Column, DateTime, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid


class TimestampMixin:
    """
    Adds created_at and updated_at fields to any model.
    SQLAlchemy handles these automatically - no manual updates needed.
    """
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class UUIDMixin:
    """
    Uses UUIDs instead of auto-incrementing integers for primary keys.
    Better for distributed systems and prevents ID guessing attacks.
    """
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )


class SoftDeleteMixin:
    """
    Allows "deleting" records without actually removing them from the database.
    Useful for audit trails and data recovery.
    """
    deleted_at = Column(DateTime, nullable=True, index=True)
    is_deleted = Column(Boolean, nullable=False, default=False, index=True)

    def soft_delete(self) -> None:
        """Mark this record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        """Bring back a deleted record"""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """
    Tracks which user created or modified a record.
    Essential for compliance and debugging.
    """
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)


def to_dict(obj: Any) -> dict:
    """
    Converts a database model to a plain Python dictionary.
    Makes it easier to serialize for API responses.
    """
    return {
        column.name: getattr(obj, column.name)
        for column in obj.__table__.columns
    }
