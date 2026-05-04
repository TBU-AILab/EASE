from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from fopimt.modul import Modul
from fopimt.modul_dto import ModulResult

from .loader_dto import ModulAPI, PackageType
from .message import Message, MessageAPI
from .message_repeating import MessageRepeating, MessageRepeatingConfig

if TYPE_CHECKING:
    from .solutions.solution import Solution


class OptimizationGoal(Enum):
    MINIMIZATION = 0
    MAXIMIZATION = 1


class TaskState(Enum):
    """
    Enum for multiprocessing states
    """

    CREATED = 0
    INIT = 1
    RUN = 2
    PAUSED = 3
    STOP = 4
    FINISH = 5
    BREAK = 6


# Basic description of configurable task modules
class TaskModulConfig(BaseModel):
    short_name: str
    parameters: dict[str, Any]  # Parameter defined by short_name and its value


class TaskConfig(BaseModel):
    name: Optional[str] = None
    author: Optional[str] = None
    max_context_size: Optional[int] = None
    feedback_from_solution: Optional[bool] = None

    initial_message: Optional[str] = None
    system_message: Optional[str] = None
    repeated_message: Optional[MessageRepeatingConfig] = None

    optimization_goal: Optional[OptimizationGoal] = None

    modules: Optional[list[TaskModulConfig]] = None


class TaskInfo(BaseModel):
    id: str | None  # uuid
    name: str | None
    date_updated: str | None
    date_created: str | None
    state: TaskState | None
    current_iteration: int | None
    iterations_valid: int | None
    iterations_invalid_consecutive: int | None
    incompatible: (
        list[list[str]] | None
    )  # list of shortnames of incompatible ModuleAPIs, always in pair
    log: list[str] | None  # error log, i.e. STATE == BREAK
    optimization_goal: Optional[OptimizationGoal] = None


class TaskData(BaseModel):
    id: str | None  # uuid
    messages: list[MessageAPI]
    solutions: list[Any]


class TaskFull(BaseModel):
    task_info: Optional[TaskInfo] = None
    task_data: Optional[TaskData] = None
    task_modules: Optional[list[ModulAPI]] = None
    task_config: Optional[TaskConfig] = None


##########################################################################
### Internal DTOs
##########################################################################


@dataclass(frozen=True)
class ModulData:
    class_ref: type[Modul]
    package_type: PackageType
    result: ModulResult | None


@dataclass(frozen=True)
class ModulInfo:
    class_ref: type[Modul]
    package_type: PackageType
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TaskExecutionContext:
    task_id: str
    task_name: str
    used_modules: list[ModulInfo]
    modules_data_by_iteration: dict[int, list[ModulData]]
    current_iteration: int
    solutions: list["Solution"]
    time_start: datetime | None
    used_tokens: int
    valid_iterations: int
    invalid_iterations: int
    system_message: Message
    initial_message: Message
    repeating_message: MessageRepeating
    optimization_goal: Optional[OptimizationGoal]
