from .stopping_condition import StoppingCondition
from ..loader import Parameter, PrimitiveType


class StoppingConditionTokens(StoppingCondition):
    """
    General class for definition of stopping conditions based on used tokens by Task.
    Arguments:
        max_tokens: int -- Number of maximal number of used tokens.
    """

    def _init_params(self):
        super()._init_params()
        self._max_tokens: int = self.parameters.get('value', 0)
        self._satisfied = False

    ####################################################################
    #########  Public functions
    ####################################################################
    def pretty(self) -> str:
        if self.is_satisfied():
            return f'Stopping condition <Tokens>: Stopped at token count at least {self._max_tokens}.'
        else:
            return f'Stopping condition <Tokens>: Not triggered at token count at least {self._max_tokens}.'

    def is_satisfied(self) -> bool:
        return self._satisfied

    def update(self, task) -> None:
        from ..task import Task
        if isinstance(task, Task):
            self._satisfied = task.get_used_tokens() >= self._max_tokens
        else:
            raise TypeError("Function update needs instance of Task")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {'value':
                    Parameter(short_name='value', long_name='Maximum number of used tokens',
                              description='Works best with OpenAI models. 0 as infinite',
                              type=PrimitiveType.int, min_value=0, max_value=999999, default=0)
                }

    @classmethod
    def get_short_name(cls) -> str:
        return "stop.condmaxtokens"

    @classmethod
    def get_long_name(cls) -> str:
        return "Maximum number of used tokens"

    @classmethod
    def get_description(cls) -> str:
        return "General class for definition of stopping conditions based on used tokens by Task."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }
    ####################################################################
    #########  Private functions
    ####################################################################
