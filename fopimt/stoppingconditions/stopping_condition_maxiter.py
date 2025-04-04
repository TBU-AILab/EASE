from .stopping_condition import StoppingCondition
from ..loader import Parameter, PrimitiveType


class StoppingConditionMaxIter(StoppingCondition):
    """
    General class for definition of stopping conditions based on the number of Task iterations.
    Arguments:
        max_iters: int  -- Maximum number of iterations.
        delta: int      -- Value of one incrementation used when called update().
    """

    def _init_params(self):
        super()._init_params()
        self._max_iters = self.parameters.get('value', 0)
        self._iters = 0
        self._delta = 1

    ####################################################################
    #########  Public functions
    ####################################################################
    def pretty(self) -> str:
        if self.is_satisfied():
            return f'Stopping condition <Iterations>: Stopped at iteration number {self._iters}.'
        else:
            return f'Stopping condition <Iterations>: Not triggered at iteration number {self._iters}.'

    def is_satisfied(self) -> bool:
        return self._iters >= self._max_iters

    def update(self, task) -> None:
        from ..task import Task
        if isinstance(task, Task):
            self._iters += self._delta
        else:
            raise TypeError("Function update needs Task")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {'value':
                    Parameter(short_name='value', long_name='Maximum number of iterations',
                              type=PrimitiveType.int, min_value=0, max_value=999999, default=1)
                }

    @classmethod
    def get_short_name(cls) -> str:
        return "stop.condmaxiters"

    @classmethod
    def get_long_name(cls) -> str:
        return "Maximum number of iterations"

    @classmethod
    def get_description(cls) -> str:
        return "General class for definition of stopping conditions based on the number of Task iterations."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }
    ####################################################################
    #########  Private functions
    ####################################################################
