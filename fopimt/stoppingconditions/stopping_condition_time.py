from .stopping_condition import StoppingCondition
from ..loader import Parameter, PrimitiveType


class StoppingConditionTime(StoppingCondition):
    """
    General class for definition of stopping conditions based on the elapsed time of the Task.
    Arguments:
        max_time: int   -- Maximum number of elapsed time in seconds.
    """

    def _init_params(self):
        super()._init_params()
        self._max_time = self.parameters.get('max_time', 0)
        self._satisfied: bool = False

    ####################################################################
    #########  Public functions
    ####################################################################

    def is_satisfied(self) -> bool:
        return self._satisfied

    def update(self, task) -> None:
        from ..task import Task
        if isinstance(task, Task):
            if task.get_time() >= self._max_time:
                self._satisfied = True
        else:
            raise TypeError("Function update needs Task")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {'max_time':
                    Parameter(short_name='max_time', long_name='Running time',
                              description='Maximum running time for Task in seconds',
                              type=PrimitiveType.time, min_value=0, max_value=31536000, default=60)
                }

    @classmethod
    def get_short_name(cls) -> str:
        return "stop.condmaxtime"

    @classmethod
    def get_long_name(cls) -> str:
        return "Maximum Task running time"

    @classmethod
    def get_description(cls) -> str:
        return "Stopping condition based on the Task running time."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }
    ####################################################################
    #########  Private functions
    ####################################################################
