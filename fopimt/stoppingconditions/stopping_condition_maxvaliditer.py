from ..loader_dto import Parameter, PrimitiveType
from ..modul_dto import StoppingConditionResult
from .stopping_condition import StoppingCondition


class StoppingConditionMaxValidIter(StoppingCondition):
    """
    Stopping condition based on the number of valid Task iterations (solutions that passed tests and evaluation).
    Arguments:
        max_iters: int  -- Maximum number of valid iterations.
    """

    def _init_params(self):
        super()._init_params()
        self._max_iters = self.parameters.get("value", 0)
        self._iters = 0

    ####################################################################
    #########  Public functions
    ####################################################################
    def pretty(self) -> str:
        result = self.is_satisfied()
        if result.is_satisfied:
            return f"Stopping condition <Valid iterations>: Stopped at valid iteration number {self._iters}."
        else:
            return f"Stopping condition <Valid iterations>: Not triggered at valid iteration number {self._iters}."

    def is_satisfied(self) -> StoppingConditionResult:
        is_satisfied = self._iters >= self._max_iters
        return StoppingConditionResult(
            class_ref=type(self),
            is_satisfied=is_satisfied,
            metadata={"iters": self._iters, "max_iters": self._max_iters},
        )

    def update(self, task) -> None:
        from ..task import Task

        if isinstance(task, Task):
            self._iters = task.get_iteration_valid()
        else:
            raise TypeError("Function update needs Task")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "value": Parameter(
                short_name="value",
                long_name="Maximum number of valid iterations",
                type=PrimitiveType.int,
                min_value=0,
                max_value=999999,
                default=1,
            )
        }

    @classmethod
    def get_short_name(cls) -> str:
        return "stop.condmaxvaliditers"

    @classmethod
    def get_long_name(cls) -> str:
        return "Maximum number of valid iterations"

    @classmethod
    def get_description(cls) -> str:
        return "Stopping condition based on the number of valid Task iterations (passed tests and evaluation)."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": set()}
