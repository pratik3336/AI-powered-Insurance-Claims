"""
User and Role models for authentication and authorization.
Implements Role-Based Access Control (RBAC) - industry standard for permissions.
"""

from sqlalchemy import Column, String, Boolean, ForeignKey, Table, Enum as SQLEnum, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin, SoftDeleteMixin


# Association table for many-to-many relationship between users and roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'))
)


class RoleType(str, enum.Enum):
    """
    Predefined roles for the system.
    Each role has different permission levels.
    """
    ADMIN = "admin"  # Full system access
    ADJUSTER = "adjuster"  # Can review and approve claims
    INVESTIGATOR = "investigator"  # Can investigate fraud cases
    VIEWER = "viewer"  # Read-only access
    API_USER = "api_user"  # Programmatic access only


class Role(Base, UUIDMixin, TimestampMixin):
    """
    Defines user roles and their permissions.
    Used for controlling access to different parts of the system.
    """
    __tablename__ = "roles"

    name = Column(
        SQLEnum(RoleType),
        unique=True,
        nullable=False,
        index=True
    )

    description = Column(String(255), nullable=True)

    # Permissions stored as JSON or separate table in production
    # For now, we'll handle permissions in application logic

    # Relationships
    users = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles"
    )

    def __repr__(self) -> str:
        return f"<Role {self.name.value}>"


class User(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    User accounts for the claims processing system.
    Supports both human users (adjusters, admins) and API users.
    """
    __tablename__ = "users"

    # Authentication
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    hashed_password = Column(
        String(255),
        nullable=False,
        comment="Bcrypt hashed password"
    )

    # Profile
    full_name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)

    # Account status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Active users can log in"
    )

    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Email verification status"
    )

    # Security
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(
        DateTime,
        nullable=True,
        comment="Account locked until this time after too many failed logins"
    )

    # Relationships
    roles = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users"
    )

    # Claims created by this user (as a claimant)
    claims = relationship(
        "Claim",
        back_populates="claimant",
        foreign_keys="Claim.claimant_id"
    )

    # Claims assigned to this user (as an adjuster)
    assigned_claims = relationship(
        "Claim",
        back_populates="assigned_adjuster",
        foreign_keys="Claim.assigned_adjuster_id"
    )

    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role"""
        return any(role.name.value == role_name for role in self.roles)

    def is_admin(self) -> bool:
        """Check if user is an administrator"""
        return self.has_role("admin")

    def can_approve_claims(self) -> bool:
        """Check if user can approve claims"""
        return self.has_role("admin") or self.has_role("adjuster")

    def __repr__(self) -> str:
        return f"<User {self.email}>"
