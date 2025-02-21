from datetime import datetime, timezone

class TaskTimeRun:
    def __init__(self):
        """
        Class for TimeStamp of the Task run.
        Contains Start time and End time.
        Automatically start the time.
        """
        self._start = datetime.now(timezone.utc)
        self._end = None
        self._duration = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def end(self) -> None:
        """
        End the time.
        """
        self._end = datetime.now(timezone.utc)
        self._duration = self._end - self._start

    def get_in_sec(self) -> str:
        return str(self._duration.seconds)

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end
    ####################################################################
    #########  Private functions
    ####################################################################


class DateTime:

    def __init__(self):
        """
        Class for DateTime handling of the Task.
        Initialization will create timestamp of the creation of the task.
        Contains information of the first creation, last used, how many times task was used, ...
        """
        self._task_created = datetime.now(timezone.utc)
        self._task_last_used = datetime.now(timezone.utc)
        self._run_history = []

    @property
    def task_created(self):
        return self._task_created

    @task_created.setter
    def task_created(self, value):
        self._task_created = value

    @property
    def run_history(self):
        return self._run_history

    @run_history.setter
    def run_history(self, value):
        self._run_history = value

    @property
    def task_last_used(self):
        return self._task_last_used

    @task_last_used.setter
    def task_last_used(self, value):
        self._task_last_used = value

    ####################################################################
    #########  Public functions
    ####################################################################
    def update_last_used(self):
        self._task_last_used = datetime.now(timezone.utc)

    def task_start(self) -> None:
        """
        Marks new start of the task.
        Adds timestamp into history of runs.
        """
        self._run_history.append(TaskTimeRun())

    def task_end(self) -> None:
        """
        Marks end of the task run.
        """
        self._run_history[-1].end()
        self._task_last_used = datetime.now(timezone.utc)

    def get_task_history(self) -> list[TaskTimeRun]:
        """
        Returns history (timestamps) of all runs of the Task.
        """
        return self._run_history

    def get_created_filename(self):
        return self._task_created.strftime("%Y%m%d-%H%M%S")

    def get_created_DateTime(self):
        return self._task_created.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


    def get_updated_DateTime(self):
        return self._task_last_used.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def convert_to_datetime(date_str: str) -> datetime:
    """
    Converts the given string to a datetime object.
    If the string is None or an empty string, returns the lowest possible datetime.
    """
    if not date_str:
        return datetime.min  # Returns the lowest possible datetime value

    try:
        # Parsing the string to datetime, assuming it is in UTC format (ending with 'Z')
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as e:
        # Handle cases where the date format is incorrect
        raise ValueError(f"Invalid date format: {e}")


def is_newer_than(date1: datetime, date2: datetime) -> bool:
    """
    Compares two datetime objects and returns True if date1 is newer than date2.
    """
    d1 = date1.replace(tzinfo=None)
    d2 = date2.replace(tzinfo=None)
    return d1 > d2

    ####################################################################
    #########  Private functions
    ####################################################################

