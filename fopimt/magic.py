import copy
import os
import logging
from typing import Union

from .utils.package_manager import PackageManager
from .loader import Loader, Package, PackageType, ModulAPI
from .task import Task, TaskState, TaskData, TaskInfo
from .message_repeating import MessageRepeatingTypeEnum, MessageRepeating
from .tests.test import Test
from .analysis.analysis import Analysis
from .evaluators.evaluator import Evaluator
from .stoppingconditions.stopping_condition import StoppingCondition

from .message import Message
from .task_manager import TaskManager

from .magic_datetime import convert_to_datetime, is_newer_than

class Magic:

    def __init__(self):
        """
        Main class of FoP-IMT.
        Contains various connectors and definitions of tasks.
        Multiprocessing of Tasks.
        """
        # Init Log
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(module)-12.12s] [%(funcName)-12.12s] [%(levelname)-5.5s]  "
                   "%(message)s",
            handlers=[
                logging.FileHandler("log.log"),
                logging.StreamHandler()
            ]
        )

        logging.info("Started")

        # App loader responsible for packages
        self._loader = Loader()

        self._package_manager = PackageManager()

        # App instance setting, pool of all Tasks
        self._tasks: dict[str, Task] = {}  # all tasks (sync with DB)

        # Sync tasks from disk
        self._init()

        # Threading & start
        self._task_manager = TaskManager()
        self._task_manager.start_monitor()
        # USAGE:RUN... probably ;-)
        # self._task_manager.add_task(task)

    ####################################################################
    #########  Public functions
    ####################################################################
    def archive(self, uid: str) -> bool:
        if uid is None or uid not in self._tasks:
            return False
        self._tasks.pop(uid)
        return True

    def get_loader(self) -> Loader:
        return self._loader

    def get_package_manager(self) -> PackageManager:
        return self._package_manager

    def get_task_info(self, uid: str) -> list[TaskInfo]:
        if uid is None or uid not in self._tasks:
            out = []
            for _task in self._tasks.values():
                out.append(_task.get_info())
            return out
        return [self._tasks[uid].get_info()]

    def get_tasks_info(self, uids: list[str]) -> list[TaskInfo]:
        if uids is None or uids == []:
            out = []
            for _task in self._tasks.values():
                out.append(_task.get_info())
            return out
        else:
            out = []
            for uid in uids:
                if uid in self._tasks:
                    out.append(self._tasks[uid].get_info())
            return out

    # Gets messages and solutions newer than a specified date
    # TODO rework Message and Solution (both probably will be based on BaseModel, or at least define their DTOs in their own classes/files)
    def get_new_data(self, date_from: str) -> list[TaskData]:
        out = []
        for task in self._tasks.values():
            if task.get_state() in [
                TaskState.FINISH,
                TaskState.RUN,
                TaskState.STOP,
                TaskState.BREAK,
                TaskState.PAUSED,
            ]:
                data = task.get_task_data(date_from)
                if len(data.solutions) != 0 or len(data.messages) != 0:
                    out.append(data)

        return out

    def task_pause(self, task_id: str) -> bool:
        if task_id == None or task_id not in self._tasks.keys():
            logging.error(f"Magic:Task Pause: Task with id({task_id}) does not exist in the system.")
            return False

        if self._tasks[task_id].state != TaskState.RUN:
            logging.error(f"Magic:Task Pause: Task with id({task_id}) is not in the RUN state.")
            return False

        result = self._task_manager.pause_task(task_id)
        if result:
            with self._task_manager.lock:
                task = self._tasks[task_id]
                task.state = TaskState.PAUSED
        return result

    def task_stop(self, task_id: str) -> bool:
        if task_id is None or task_id not in self._tasks.keys():
            logging.error(f"Magic:Task Stop: Task with id({task_id}) does not exist in the system.")
            return False

        if self._tasks[task_id].state not in [TaskState.RUN, TaskState.PAUSED]:
            logging.error(f"Magic:Task Stop: Task with id({task_id}) is not in the RUN state.")
            return False

        result = self._task_manager.stop_task(task_id)
        if result:
            with self._task_manager.lock:
                task = self._tasks[task_id]
                task.state = TaskState.STOP
        return result

    def task_finished_callback(self, task: Task):
        self._tasks[task.id] = task

    def task_run(self, uid: str) -> bool:
        # 1) Check existence of the task and if not init return False
        # 2) If task is paused, resume it otherwise run new task in separate thread
        # ------------------------------------------------------
        # 1)
        if uid in self._tasks.keys():
            task = self._tasks[uid]

            if task.state != TaskState.INIT and task.state != TaskState.PAUSED:
                logging.info(f"Magic:Task Run: Task with id({uid}) is not fully initialized.")
                return False
        else:
            logging.info(f"Magic:Task Run: Task with id({uid}) does not exist in the system.")
            return False            

        # 2)
        try:
            if task.state == TaskState.PAUSED:
                task.state = TaskState.RUN
                return self._task_manager.resume_task(uid)
                
            task.state = TaskState.RUN
            self._task_manager.add_task(task, callback=self.task_finished_callback)
        except Exception as e:
            logging.error(f"Magic:Task Run: Task with id({uid}) could not be run due to: repr{e}")
            task.state = TaskState.BREAK
            return False

        return True

    def task_get_all(self) -> list[Task]:
        task_list = []
        for uuid in self._tasks.keys():
            task_list.append(self._tasks[uuid])
        return task_list

    def task_get(self, uid: str) -> Task | None:
        if uid in self._tasks.keys():
            return self._tasks[uid]
        else:
            logging.error(f'Task with ID:{uid} does not exist.')
            return None

    def task_create(self, uid: str = None) -> Union[Task|None]:
        # get config ID of a task
        task = Task.create_empty(task_id=uid)
        if task is None:
            return None
        # what if task already exists in pool? throw error? or just return existing task? DUNNO
        if task.id in self._tasks.keys():
            logging.warning(f'Task with ID:{task.id} already exists.')
            return self._tasks[task.id]
        self._tasks[task.id] = task

        return task

    def task_duplicate(self, orig_task: Task, new_name: str) -> Task:
        if not hasattr(orig_task, "_init_config"):
            logging.error(f"Error while duplicating: Task with ID:{orig_task.id} does not have a valid _init_config attribute.")
            return None

        # If task is not initialized, return a clean task
        if orig_task._init_config is None:
            logging.warning(f"Task with ID:{orig_task.id} does not have a valid _init_config attribute. Duplicating as a clean task.")
            new_task = self.task_create()
            new_task._name = new_name
            return new_task

        logging.info(f"Duplicating task with ID:{orig_task.id} to new task with name:{new_name}.")
        new_task = self.task_create()
        new_task.initialize(self._loader, orig_task._init_config)
        new_task._name = new_name
        return new_task


    def get_llm_connectors(self) -> list[ModulAPI]:
        return self._loader.get_package(PackageType.LLMConnector).get_moduls()

    ####################################################################
    #########  Private functions
    ####################################################################

    def _init(self) -> None:
        # Check the existence of the out_task folder
        self._dir_tasks = 'out_task'
        if not os.path.exists(self._dir_tasks):
            os.makedirs(self._dir_tasks)
            # Nothing to load, return to __init__
            logging.info('No previous tasks founded.')
            return
        # Load all folders from parent folder and Filter only valid tasks. (starts with t_)
        task_folders = [item for item in os.listdir(self._dir_tasks)
                        if os.path.isdir(os.path.join(self._dir_tasks, item)) and item.startswith('t_')
                        ]
        logging.info(f'Found {len(task_folders)} tasks.')

        # create instance from the folder
        for folder in task_folders:
            try:
                _task: Task = Task.pickle_rick(folder)
                self._tasks[_task.id] = _task
            except Exception as e:
                logging.error(f'Unable to create instance of Task from storage:{folder}. {e}')
