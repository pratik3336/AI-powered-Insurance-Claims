"""
Workflow state model - tracks LangGraph execution history.
"""

from sqlalchemy import Column, String, ForeignKey, JSON, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class WorkflowState(Base, UUIDMixin, TimestampMixin):
    """
    Tracks each step in the LangGraph workflow execution.
    Provides audit trail and debugging for automated decisions.
    """
    __tablename__ = "workflow_states"

    claim_id = Column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Workflow execution tracking
    node_name = Column(
        String(100),
        nullable=False,
        comment="Which workflow node was executed"
    )

    sequence_number = Column(
        Integer,
        nullable=False,
        comment="Order of execution in workflow"
    )

    # Node execution details
    input_state = Column(
        JSON,
        nullable=True,
        comment="State when entering this node"
    )

    output_state = Column(
        JSON,
        nullable=True,
        comment="State when exiting this node"
    )

    decision = Column(
        String(100),
        nullable=True,
        comment="Decision made by this node"
    )

    reasoning = Column(
        Text,
        nullable=True,
        comment="Why this decision was made (from LLM)"
    )

    # Performance tracking
    execution_time_ms = Column(
        Integer,
        nullable=True,
        comment="How long this node took to execute"
    )

    # Error handling
    error = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Relationships
    claim = relationship("Claim", back_populates="workflow_states")

    def __repr__(self) -> str:
        return f"<WorkflowState {self.node_name} - {self.decision}>"
