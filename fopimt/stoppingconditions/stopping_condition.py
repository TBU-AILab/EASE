from ..modul import Modul


class StoppingCondition(Modul):
    """
    General class for definition of stopping conditions.
    """

    def _init_params(self):
        super()._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    def pretty(self) -> str:
        """
        Function that will return string suitable for output in pretty format.
        e.g. 'Stopping condition <Iterations>: Stopped at 5th iteration.'
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def is_satisfied(self) -> bool:
        """
        Function that will return bool value if stopping condition was satisfied.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def update(self, task) -> None:
        """
        Function that will update and check if the stopping condition was satisfied.
        Arguments:
            task: Task  -- Instance of the parent Task.
        """
        from ..task import Task
        if isinstance(task, Task):
            raise NotImplementedError("This method should be overridden by subclasses.")
        else:
            raise TypeError("Function update needs Task")
    ####################################################################
    #########  Private functions
    ####################################################################
