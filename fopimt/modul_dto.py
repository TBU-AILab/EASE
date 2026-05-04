from typing import Any

from pydantic import BaseModel, Field

from .message import Message
from .modul import Modul


class ModulResult(BaseModel):
    class_ref: type["Modul"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluatorResult(ModulResult):
    fitness: float


class LLMConnectorResult(ModulResult):
    model_config = {"arbitrary_types_allowed": True}
    response: Message


class AnalysisResult(ModulResult):
    pass


class StatResult(ModulResult):
    pass


class StoppingConditionResult(ModulResult):
    is_satisfied: bool


class TestResult(ModulResult):
    passed: bool


class SolutionResult(ModulResult):
    evaluator_input: Any
    evaluator_input_serialized: str  # Used for rendering
