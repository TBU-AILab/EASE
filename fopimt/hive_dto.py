import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .task_dto import OptimizationGoal, TaskState

# Special recipient/sender identifiers used in HiveMessage routing
RECIPIENT_BROADCAST = "broadcast"
RECIPIENT_HIVE = "hive"
SENDER_HIVE = "hive"
SENDER_USER = "user"


class HiveState(Enum):
    """
    Lifecycle states of a Hive. Values mirror TaskState for API consistency.
    """

    CREATED = 0
    INIT = 1
    RUN = 2
    PAUSED = 3
    STOP = 4
    FINISH = 5
    BREAK = 6


class HiveMessageType(Enum):
    """
    Types of messages exchanged over the Hive bus.
    REQUEST, SPAWN and DESPAWN are reserved for phase 2 (subtask delegation
    and dynamic member management) - the orchestrator only logs them for now.
    """

    INFO = 0
    ITERATION_REPORT = 1
    REQUEST = 2
    SPAWN = 3
    DESPAWN = 4


class HivePolicyType(Enum):
    """
    Built-in routing policies of the Hive orchestrator.
    - MANUAL: no automatic routing, only explicitly addressed messages
    - BROADCAST_BEST: new globally best solution is sent to all other members
    - RING: iteration results are sent to the next member in the ring
    - RANDOM: iteration results are sent to one random other member
    - RANDOM_POOL: iteration results are sent to `pool_size` random members
    """

    MANUAL = 0
    BROADCAST_BEST = 1
    RING = 2
    RANDOM = 3
    RANDOM_POOL = 4


class HiveMessage(BaseModel):
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str  # task id | 'hive' | 'user'
    recipient: str  # task id | 'broadcast' | 'hive'
    msg_type: HiveMessageType = HiveMessageType.INFO
    payload: str = ""  # phase 1: text only
    metadata: Optional[dict[str, Any]] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )


class HiveMessageCreate(BaseModel):
    """
    API input for manual message injection (POST /hive/{hive_id}/message).
    """

    recipient: str = RECIPIENT_BROADCAST  # task id | 'broadcast'
    payload: str


class HiveStoppingConfig(BaseModel):
    """
    Global (hive-level) stopping conditions evaluated by the orchestrator
    over all members combined. Any satisfied condition stops the whole Hive.
    """

    max_total_iterations: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_time_seconds: Optional[int] = None


class HiveConfig(BaseModel):
    name: Optional[str] = None
    member_task_ids: list[str] = []
    policy: HivePolicyType = HivePolicyType.MANUAL
    policy_params: dict[str, Any] = {}  # e.g. {'pool_size': 2} for RANDOM_POOL
    optimization_goal: Optional[OptimizationGoal] = OptimizationGoal.MINIMIZATION
    stopping: Optional[HiveStoppingConfig] = None


class HiveInfo(BaseModel):
    id: str | None
    name: str | None
    state: HiveState | None
    policy: HivePolicyType | None
    member_ids: list[str]
    member_states: dict[str, TaskState] = {}
    messages_routed: int = 0
    date_created: str | None = None
    date_updated: str | None = None
    stop_reason: Optional[str] = None
    log: list[str] = []  # error log, i.e. STATE == BREAK
