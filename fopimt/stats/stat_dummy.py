from fopimt.solutions.solution import Solution
from fopimt.stats.stat import Stat, StatResult
from fopimt.task_dto import TaskExecutionContext


class StatDummy(Stat):
    def _init_params(self):
        super()._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(
        self,
        solutions: list[Solution],
        task_execution_context: TaskExecutionContext,
    ) -> StatResult:
        # self._task_execution_context = task_execution_context

        return StatResult(class_ref=type(self), metadata={"dummy": "value"})

    def export(self, path: str) -> None:
        return

    @classmethod
    def get_short_name(cls) -> str:
        return "stat.dummy"

    @classmethod
    def get_long_name(cls) -> str:
        return "Dummy Statistic"

    @classmethod
    def get_description(cls) -> str:
        return "Dummy statistic for testing purposes. Does not perform any actual evaluation."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"html"}}

    ####################################################################
    #########  Private functions
    ####################################################################
