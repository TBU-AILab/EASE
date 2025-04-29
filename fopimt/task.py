import copy
import csv
import io

import logging

import os
import pathlib
import re
import uuid
import pickle
from datetime import datetime, timezone
from enum import Enum
from queue import Queue
from typing import Optional, Any

from pandas.core.arrays.period import raise_on_incompatible
from pydantic import BaseModel

from .tests.test import Test
from .user import User
from .evaluators.evaluator import Evaluator
from .magic_datetime import DateTime, is_newer_than, convert_to_datetime
from .message import Message, MessageAPI
from .message_repeating import MessageRepeating, MessageRepeatingConfig
from .llmconnectors.llmconnector import LLMConnector
from .stoppingconditions.stopping_condition import StoppingCondition
from .solutions.solution import Solution, SolutionAPI
from .analysis.analysis import Analysis
from .stats.stat import Stat
from .loader import Loader, PackageType, ModulAPI
from .utils.tools import get_zip_buffer as zip_it
from .config_task import ConfigTask

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
    incompatible: list[list[str]] | None # list of shortnames of incompatible ModuleAPIs, always in pair
    log: list[str] | None # error log, i.e. STATE == BREAK


class TaskData(BaseModel):
    id: str | None  # uuid
    messages: list[MessageAPI]
    solutions: list[SolutionAPI]

class TaskFull(BaseModel):
    task_info: Optional[TaskInfo] = None
    task_data: Optional[TaskData] = None
    task_modules: Optional[list[ModulAPI]] = None


class Task():
    @classmethod
    def pickle_rick(cls, task_folder: str):
        _file_name = os.path.join('out_task', task_folder, 'task.pkl')
        with open(_file_name, 'rb') as _file:
            return pickle.load(_file)


    def initialize(self, loader: Loader, task_config: TaskConfig):
        if task_config.name is not None:
            self._name = task_config.name
        if task_config.author is not None:
            self._author = task_config.author
        if task_config.max_context_size is not None:
            self._max_context_size = task_config.max_context_size
        if task_config.feedback_from_solution is not None:
            self._spec_feedback_from_solution = task_config.feedback_from_solution

        # Moduls
        for modul in task_config.modules:
            _response = loader.get_modul_by_name(short_name=modul.short_name)
            if _response is not None:
                _modul, _type = _response
                self._init_config_modulAPI.append(loader.get_package(_type).get_modul(modul.short_name))
                match _type:
                    case PackageType.LLMConnector:
                        self._spec_llm = _modul(parameters=modul.parameters)
                    case PackageType.Evaluator:
                        self._spec_evaluator = _modul(parameters=modul.parameters)
                    case PackageType.Solution:
                        self._spec_solution = _modul(parameters=modul.parameters)
                    case PackageType.Test:
                        self._spec_test.append(_modul(parameters=modul.parameters))
                    case PackageType.Analysis:
                        self._spec_analysis.append(_modul(parameters=modul.parameters))
                    case PackageType.StoppingCondition:
                        self._spec_cond.append(_modul(parameters=modul.parameters))
                    case PackageType.Stat:
                        self._spec_stat.append(_modul(parameters=modul.parameters))

        model = self._spec_llm.get_model() if self._spec_llm is not None else 'gpt-4o'

        if task_config.system_message is not None:
            role = self._spec_llm.get_role_system() if self._spec_llm is not None else 'undefined'
            self._spec_system_message = Message(role=role,
                                                message=task_config.system_message,
                                                model_encoding=model)

        if task_config.initial_message is not None:
            role = self._spec_llm.get_role_user() if self._spec_llm is not None else 'undefined'
            self._spec_init_message = Message(role=role,
                                              message=task_config.initial_message,
                                              model_encoding=model)

        if task_config.repeated_message is not None:
            role = self._spec_llm.get_role_user() if self._spec_llm is not None else 'undefined'
            self._spec_rep_message = MessageRepeating(msg_type=task_config.repeated_message.type, role=role,
                                                      model_encoding=model, msgs=task_config.repeated_message.msgs,
                                                      weights=tuple(task_config.repeated_message.weights))

        # Chceck if the task can be sucessfuly initialized
        if not self.is_init():
            logging.error("Task:run: Task is not completely defined.")
            raise AttributeError("There was an error during task initialization. Perhaps some task data are missing.")

        self._init_config = task_config
        self.pickle_me()


    # A class method to create an empty Task
    @classmethod
    def create_empty(cls, task_id: str):
        """Creates a new empty - uninitialized Task object."""
        # create an empty task with dummy values
        # TODO: Think about what these default values should be
        config = {
            ConfigTask.AUTHOR: '',
            ConfigTask.NAME: '',
            ConfigTask.MAX_CONTEXT_SIZE: -1,
            ConfigTask.SAVE_TO_DISK: True,
            ConfigTask.FEEDBACK_FROM_SOLUTION: True,
            ConfigTask.SYSTEM_MESSAGE: None,
            ConfigTask.INIT_MESSAGE: None,
            ConfigTask.LLM: None,
            ConfigTask.SOLUTION: None,
            ConfigTask.TEST: [],
            ConfigTask.ANALYSIS: [],
            ConfigTask.EVALUATOR: None,
            ConfigTask.COND: [],
            ConfigTask.STAT: [],
            ConfigTask.REP_MESSAGE: None
        }

        task = cls(config)
        task._state = TaskState.CREATED
        if task_id is not None:
            task._id = task_id
        if not task.create_dir():
            return None
        task.pickle_me()
        return task

    # TODO Legacy __init__
    def __init__(self, config: dict):
        """
        Task class. Needs to be initialized before run using conf.
        Use ConfigTask(Enum) for proper initialization.
        Task contains all information including history, initialization, LLM type, author, timestamps, and many more.
        Arguments:
            config: dict    -- Configuration dictionary. Use keywords from ConfigTask(Enum).
        """
        if config is None or not config:
            raise AttributeError("Missing config for Task")

        for key in ConfigTask:
            if key not in config.keys():
                raise KeyError(f"Missing {key} in Task config")

        # TODO @Adam add cond that at least one condition is present and it is iteration (default one iteration)

        # Task descriptors
        self._id: str = str(uuid.uuid4())  # TODO Unique ID - check if unique in DB
        self._name: str = config[ConfigTask.NAME]  # Custom name
        self._author: User = config[ConfigTask.AUTHOR]  # Author - instance of Author?
        self._date: DateTime = DateTime()  # Datetime - created, last modified, last run, all runs timestamps
        self._state: TaskState = TaskState.INIT

        # Task properties
        self._iteration: int = 0  # Used iterations so far
        self._iteration_valid: int = 0  # Valid iterations of the Task (state == 'OK')
        self._iteration_invalid_cons: int = 0  # Consecutive invalid iterations of the Task (state != 'OK')
        self._history_message: list[Message] = []  # History of all Messages send and received
        self._history_solution: list[Solution] = []  # History of all Solutions
        self._max_context_size: int = config[
            ConfigTask.MAX_CONTEXT_SIZE]  # Number of messages to be sent (messages in context)
        self._dir: str = os.path.join('out_task')  # Path to main task directory
        self._dir_solution: str = ''
        self._dir_stat: str = ''
        self._dir_anal: str = ''
        self._members: list[User] = []  # Additional members that can view and modify this task, can only be set by
        self._incompatible_modules = []
        self._log_error = []
        self._init_config: TaskConfig = None # Original TaskConfig for easy duplication
        self._init_config_modulAPI: list[ModulAPI] = [] # Saved ModulAPIs for Task synchronization on server start
        # Author of the Task #TODO redo to config

        # Task specifications
        self._spec_system_message: Optional[Message] = config[
            ConfigTask.SYSTEM_MESSAGE]  # Initial code/prompt, using Message class, may be empty if needed
        self._spec_init_message: Message = config[ConfigTask.INIT_MESSAGE]  # Initial code/prompt, using Message class
        self._spec_rep_message: Message = config[ConfigTask.REP_MESSAGE]  # Repetitive Message definition
        self._spec_llm: LLMConnector = config[ConfigTask.LLM]  # LLM definition - connector
        self._spec_test: list[Test] = config[ConfigTask.TEST]  # Errors definition / unit test / static code analysis
        self._spec_analysis: list[Analysis] = config[ConfigTask.ANALYSIS]  # Analysis tools
        self._spec_evaluator: Evaluator = config[
            ConfigTask.EVALUATOR]  # Evaluator object capable of solution fitness evaluation
        self._spec_cond: list[StoppingCondition] = config[ConfigTask.COND]  # Stopping condition(s)
        self._spec_res = None  # Results (+ visuals)
        self._spec_solution: Solution = config[ConfigTask.SOLUTION]  # Solution of the Task
        self._spec_save_to_disk: bool = config[ConfigTask.SAVE_TO_DISK]  # Option: save task files to disk
        self._spec_feedback_from_solution: bool = config[ConfigTask.FEEDBACK_FROM_SOLUTION]  # Option: send feedback
        # to LLM from solution
        self._spec_stat: list[Stat] = config[ConfigTask.STAT]  # Optional Statistics

        # Set encoding of Messages based on the LLMConnector.model
        # Only if specified LLM is set
        if self._spec_llm is not None:
            self._spec_init_message.set_model(self._spec_llm.get_model())
            self._spec_rep_message.set_model(self._spec_llm.get_model())
            if self._spec_system_message is not None:
                self._spec_system_message.set_model(self._spec_llm.get_model())

        # TODO remove spec from config
        # Save to disk is always true on the server
        self._spec_save_to_disk = True


    def get_all(self) -> io.BytesIO | None:
        return zip_it(self._dir)

    def get_solution(self, message_id: str) -> io.BytesIO | None:
        for sol in self._history_solution:
            if sol.get_message_id() == message_id:
                return zip_it(os.path.dirname(sol.get_path()))
        return None

    def archive(self) -> bool:
        if self._state in [TaskState.RUN, TaskState.PAUSED]:
            return False
        # rename the directory. keep everything ales unchanged, even self._dir. So, restoring is only about renaming
        # the directory of the task. Be carefully for duplicit ID when restoring or archiving task.
        # TODO - check if archived task with the same ID already exist
        old_dir = os.path.basename(self._dir)
        new_dir = old_dir.replace("t_", "a_", 1) if old_dir.startswith("t_") else old_dir
        os.rename(self._dir, os.path.join('out_task', new_dir))
        logging.info(f'Archived Task:{self._name}:{self._id}')
        return True

    def get_task_data(self, date_from: str) -> TaskData:
        msgs = []
        sols = []

        # Task Role provider for MessageAPI
        _roleProvider = [
            self._spec_llm.get_role_system(),
            self._spec_llm.get_role_user(),
            self._spec_llm.get_role_assistant()
        ]

        datefrom_dt = convert_to_datetime(date_from)
        for msg in self._history_message:
            if is_newer_than(msg.get_timestamp(), datefrom_dt):
                msgs.append(msg.get_API(_roleProvider))
                for sol in self._history_solution:
                    if sol.get_message_id() == msg.get_id():
                        sols.append(sol.get_API())

        return TaskData(
            id=self._id,
            messages=msgs,
            solutions=sols
        )

    def get_info(self):
        return TaskInfo(
            id=self._id,
            name=self._name,
            date_updated=self._date.get_created_DateTime(),
            date_created=self._date.get_created_DateTime(),
            state=self._state,
            current_iteration=self._iteration,
            iterations_valid=self._iteration_valid,
            iterations_invalid_consecutive=self._iteration_invalid_cons,
            incompatible=self._incompatible_modules,
            log=self._log_error
        )

    def get_full(self):
        modules = copy.deepcopy(self._init_config_modulAPI)
        for module in modules:
            modul_with_value = next(x for x in self._init_config.modules if x.short_name == module.short_name)
            for key in module.parameters.keys():
                module.parameters[key].value = modul_with_value.parameters[key]

        return TaskFull(
            task_info=self.get_info(),
            task_data=self.get_task_data("0001-01-01T00:00:00.000000Z"), #I want to load all messages
            task_modules=modules
        )

    # TODO clean setters/getters
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def members(self):
        return self._members

    @members.setter
    def members(self, value):
        self._members = value

    @property
    def spec_res(self):
        return self._spec_res

    @spec_res.setter
    def spec_res(self, value):
        self._spec_res = value

    @property
    def spec_evaluator(self):
        return self._spec_evaluator

    @spec_evaluator.setter
    def spec_evaluator(self, value):
        self._spec_evaluator = value

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, value):
        self._time_end = value

    @property
    def history_solution(self):
        return self._history_solution

    @history_solution.setter
    def history_solution(self, value):
        self._history_solution = value

    @property
    def dir_solution(self):
        return self._dir_solution

    @dir_solution.setter
    def dir_solution(self, value):
        self._dir_solution = value

    @property
    def history_message(self):
        return self._history_message

    @history_message.setter
    def history_message(self, value):
        self._history_message = value

    @property
    def spec_cond(self):
        return self._spec_cond

    @spec_cond.setter
    def spec_cond(self, value):
        self._spec_cond = value

    @property
    def spec_save_to_disk(self):
        return self._spec_save_to_disk

    @spec_save_to_disk.setter
    def spec_save_to_disk(self, value):
        self._spec_save_to_disk = value

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def max_context_size(self):
        return self._max_context_size

    @max_context_size.setter
    def max_context_size(self, value):
        self._max_context_size = value

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, value):
        self._time_start = value

    @property
    def iteration_valid(self):
        return self._iteration_valid

    @iteration_valid.setter
    def iteration_valid(self, value):
        self._iteration_valid = value

    @property
    def spec_stat(self):
        return self._spec_stat

    @spec_stat.setter
    def spec_stat(self, value):
        self._spec_stat = value

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, value):
        self._dir = value

    @property
    def spec_rep_message(self):
        return self._spec_rep_message

    @spec_rep_message.setter
    def spec_rep_message(self, value):
        self._spec_rep_message = value

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = value

    @property
    def spec_feedback_from_solution(self):
        return self._spec_feedback_from_solution

    @spec_feedback_from_solution.setter
    def spec_feedback_from_solution(self, value):
        self._spec_feedback_from_solution = value

    @property
    def spec_system_message(self):
        return self._spec_system_message

    @spec_system_message.setter
    def spec_system_message(self, value):
        self._spec_system_message = value

    @property
    def dir_stat(self):
        return self._dir_stat

    @property
    def dir_anal(self):
        return self._dir_anal

    @dir_anal.setter
    def dir_anal(self, value):
        self._dir_anal = value

    @dir_stat.setter
    def dir_stat(self, value):
        self._dir_stat = value

    @property
    def spec_init_message(self):
        return self._spec_init_message

    @spec_init_message.setter
    def spec_init_message(self, value):
        self._spec_init_message = value

    @property
    def iteration_invalid_cons(self):
        return self._iteration_invalid_cons

    @iteration_invalid_cons.setter
    def iteration_invalid_cons(self, value):
        self._iteration_invalid_cons = value

    @property
    def spec_solution(self):
        return self._spec_solution

    @spec_solution.setter
    def spec_solution(self, value):
        self._spec_solution = value

    @property
    def spec_test(self):
        return self._spec_test

    @spec_test.setter
    def spec_test(self, value):
        self._spec_test = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def author(self):
        return self._author

    @author.setter
    def author(self, value):
        self._author = value

    @property
    def spec_llm(self):
        return self._spec_llm

    @spec_llm.setter
    def spec_llm(self, value):
        self._spec_llm = value

    @property
    def spec_analysis(self):
        return self._spec_analysis

    @spec_analysis.setter
    def spec_analysis(self, value):
        self._spec_analysis = value

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    def task_config(self) -> TaskConfig:
        return self._init_config



    ####################################################################
    #########  Public functions
    ####################################################################
    def pickle_me(self) -> None:
        # Create the file path
        file_path = os.path.join(self._dir, 'task.pkl')
        try:
            # Open the file without a context manager for more control
            file = open(file_path, 'wb')
            try:
                pickle.dump(self, file)  # Replace with the data you want to pickle
            except Exception as e:
                logging.error(f"Error during pickling: {e}")
            finally:
                file.close()
        except Exception as e:
            logging.error(f"Error opening file {file_path} for pickling: {e}")

    def run(self):
        """
        Run the Task. Will test if all requirements are defined.
        Returns Solution.
        """
        self._log_error = []
        if not self.is_init():
            logging.error("Task:run: Task is not completely defined.")

        # TODO add optional ON/OFF option for debug and info messages

        self._date.task_start()
        logging.info(f'Task[{self._id}] start | time: {self._date.get_task_history()[-1].get_start()}')

        # calling the true 'iterative' run
        try:
            self._state = TaskState.RUN
            self._run()
        except Exception as e:
            logging.error('Unexpected error during Task run (BREAKING):', repr(e))
            self._log_error.append(f'Unexpected error during Task run (BREAKING): {repr(e)}')
            self._state = TaskState.BREAK
            self.pickle_me()
            return self

        self._date.task_end()

        logging.info(f'Task[{self._id}] finish | time: {self._date.get_task_history()[-1].get_in_sec()} s')
        self._state = TaskState.FINISH
        self.pickle_me()

        return self

    def get_used_tokens(self) -> int:
        """
        Returns the number of all used tokens. Will most probably contain real and expected tokens.
        """
        # TODO add proper variable that will store the information about used tokens from LLMConnector.
        token_count = 0
        for msg in self._history_message:
            token_count += msg.get_tokens()
        return token_count

    def get_state(self) -> TaskState:
        """
        Returns the state of the Task
        :return: (TaskState)
        """
        return self._state

    def get_id(self) -> str:
        """
        Gets ID of the Task
        :return: (str)
        """
        return self._id

    def create_dir(self) -> bool:
        try:
            self._dir = os.path.join(self._dir, 't_' + self._id)
            os.mkdir(self._dir)
        except Exception as e:
            logging.error(f'Task:{self._name}:{self._id} cant create folder! Exception: {e}')
            return False

        # create sub-folders and save path
        self._dir_solution = os.path.join(self._dir, 'solution')
        self._dir_stat = os.path.join(self._dir, 'stat')
        self._dir_anal = os.path.join(self._dir, 'anal')
        os.mkdir(self._dir_solution)
        os.mkdir(self._dir_stat)
        os.mkdir(self._dir_anal)
        return True

    def get_iteration_valid(self) -> int:
        """
        Returns the number of valid iterations.
        :return: int
        """
        return self._iteration_valid

    def get_iteration_invalid_cons(self) -> int:
        """
        Returns the number of consecutive invalid iterations.
        :return: int
        """
        return self._iteration_invalid_cons

    def get_time(self) -> int:
        start = self._date.get_task_history()[-1].get_start()
        end = datetime.now(timezone.utc)
        duration = end - start
        return duration.seconds

    ####################################################################
    #########  Private functions
    ####################################################################
    def _get_message(self, state: str, queue: Queue) -> Message:
        """
        Internal function to get Message based on the passed state.
        Arguments:
            state: str  -- State of the internal _run().
        """
        if state == 'INIT':
            return self._spec_init_message

        msg_str = ""
        if state == 'OK':
            msg_str += self._spec_rep_message.get_content() + "\n"

        while not queue.empty():
            msg_str += queue.get()
            msg_str += '\n'

        return Message(model_encoding=self._spec_llm.get_model(), message=msg_str, role=self._spec_llm.get_role_user())

    def _add_to_history(self, msg: Message) -> None:
        """
        Internal function to save Message to history.
        Save the message to disk if set by user.
        Arguments:
            msg: Message    -- Message object to be saved into history.
        """
        self._history_message.append(msg)

        if self._spec_save_to_disk:
            _path = os.path.join(self._dir, 'messages.csv')
            _init = os.path.isfile(_path)
            with open(_path, 'a', encoding='utf-8', newline='') as csv_file:
                fieldnames = ['time', 'tokens', 'role', 'text', 'path_solution_file', 'path_solution_metadata', 'id',
                              'metadata']
                wr = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
                if not _init:
                    wr.writeheader()
                ret = self._get_solution_paths(msg.get_id())
                wr.writerow({'time': msg.get_timestamp(), 'tokens': msg.get_tokens(),
                             'role': msg.get_role(), 'text': str(msg.get_content()),
                             'id': msg.get_id(),
                             'path_solution_file': ret.get('path_solution_file', None),
                             'path_solution_metadata': ret.get('path_solution_metadata', None),
                             'metadata': msg.get_metadata()
                             })
                csv_file.close()

    def _get_solution_paths(self, id: str) -> dict:
        ret = {}
        for sol in self._history_solution:
            if sol.get_message_id() == id:
                ret['path_solution_file'] = sol.get_path()
                ret['path_solution_metadata'] = sol.get_path_metadata()
                break
        return ret

    def _get_context(self) -> list[Message]:
        """
        Internal function to make context to send to LLM.
        Returns list of Messages. Length and context is specified by settings of the Task.
        """
        if self._max_context_size == 0:
            if self._spec_system_message is not None:
                return [self._spec_system_message]
            else:
                # return at least init message
                return [self._spec_init_message]

        # unlimited context size or no slicing needed
        if self._max_context_size < 0 or self._max_context_size > len(self._history_message):
            cnx = copy.deepcopy(self._history_message)
            return cnx

        # limited context size
        if self._spec_system_message is not None:
            cnx = [self._spec_system_message]
        else:
            cnx = []
        cnx += copy.deepcopy(self._history_message[-self._max_context_size - 1:])
        return cnx

    def _run(self) -> None:
        """
        Internal iterative run of the Task.
        """

        # save task time
        self._time_start = datetime.now(timezone.utc)

        state = 'INIT'
        if self._spec_system_message is not None:
            self._spec_system_message.update_timestamp()
            self._history_message = []
            self._add_to_history(self._spec_system_message)
        else:
            self._history_message = []
        buffer_message = Queue(maxsize=0)  # Buffer (queue) of messages to be sent to LLM in new iteration

        if self._spec_init_message is not None:
            self._spec_init_message.update_timestamp()

        while (state != 'STOP'):
            # 1) get context
            # 2) add new message to history and to context
            # 3) pass context to LLM
            # 4) get response from LLM
            # 5) add response to history
            # 6) check for errors/unit testing
            # 7) if state of the errors is 'OK' (i.e. no errors) GOTO 9)
            # 8) if state of the errors is 'ERROR' GOTO 1)
            # 9) check for optional analysis
            # 10) evaluation
            # 11) check stopping conditions
            # 12) if state of the condition is 'OK' (i.e. no stopping) GOTO 1)
            # 13) if state of the condition is 'STOP' GOTO 14)
            # 14) End and return solution

            #####################################################################

            # 1) get context
            context = self._get_context()

            # 2) add new message to history and to context
            # 2) loop over queue of strings, convert them to one User message and send to LLM.
            # Also, add to history and context
            # And save the message to disk
            msg = self._get_message(state, buffer_message)
            context.append(msg)
            self._add_to_history(msg)

            # 3) pass context to LLM & # 4) get response from LLM
            response = self._spec_llm.send(context)

            # 5) create a solution
            solution = copy.deepcopy(self._spec_solution)
            solution.get_input_from_msg(response)
            solution.set_message_id(response.get_id())

            # 5.5) add response to history
            self._add_to_history(response)

            # export solution
            if self._spec_save_to_disk:
                sol_dir = os.path.join(self._dir_solution, 'sol_' + str(self._iteration))
                os.mkdir(sol_dir)
                solution.export(dir=sol_dir, id='sol_' + str(self._iteration))

            # 6) check for errors/unit testing
            state = 'OK'
            for test in self._spec_test:
                if not test.test(solution):
                    logging.error(f'Task[{self._id}]: {test.get_user_msg()}')
                    buffer_message.put(test.get_error_msg())
                    state = 'ERROR'
                else:
                    logging.info(f'Task[{self._id}]: {test.get_user_msg()}')

            # 7) if state of the errors is 'OK' (i.e. no errors) GOTO 9)
            # 8) if state of the errors is 'ERROR' GOTO 1)

            # 9) check for optional analysis
            if state == 'OK':
                for anal in self._spec_analysis:
                    anal.evaluate_analysis(solution)
                    anal.export(path=self._dir_anal, id='anal_' + str(self._iteration))

            # 10) evaluation, only if no ERROR
            if state == 'OK':
                self._spec_evaluator.evaluate(solution)
                if self._spec_feedback_from_solution:
                    buffer_message.put(solution.get_feedback())
                    for anal in self._spec_analysis:
                        buffer_message.put(anal.get_feedback())

            # export metadata of evaluated solution
            if self._spec_save_to_disk:
                solution.export_meta()

            # copy so the saved solution in history does not change
            self._history_solution.append(copy.deepcopy(solution))

            # 11) check stopping conditions
            self._iteration += 1
            if state == 'OK':
                self._iteration_valid += 1
                self._iteration_invalid_cons = 0
            # invalid iteration counter
            else:
                self._iteration_invalid_cons += 1
            for cond in self._spec_cond:
                cond.update(self)
                if cond.is_satisfied():
                    state = 'STOP'

            # 12) if state of the condition is 'OK' (i.e. no stopping) GOTO 1)
            # EMPTY

            # 13) if state of the condition is 'STOP' GOTO 14)
            # EMPTY

            pass  # end while

        # 14) End and return solution
        self._spec_solution = self._spec_evaluator.get_best()

        # 14.5) Optional statistic
        for stat in self._spec_stat:
            stat.evaluate_statistic(solutions=self._history_solution)
            stat.export(self._dir_stat)

        # save end time
        self._time_end = datetime.now(timezone.utc)

    def is_init(self) -> bool:
        """
        Test if all requirements are defined.
        Should probably check the instances of tests, analysis, and conditions, because they are defined as lists.
        But for now check only if stopping conditions are not empty. If so, return False.
        Returns Bool.
        """
        self._state = TaskState.CREATED

        # LLM
        if self._spec_llm is None:
            return False

        # Repeated message
        if self._spec_rep_message is None:
            return False

        # Evaluator
        if self._spec_evaluator is None:
            return False

        # Stopping conditions
        if self._spec_cond is None or len(self._spec_cond) == 0:
            return False

        # Other parameters
        if self._name is None:
            return False
        if self._max_context_size is None:
            return False
        # if self._spec_system_message is None:
        #     return False
        if self._spec_init_message is None:
            return False

        # Set task state to INIT since all requirements are met
        self._state = TaskState.INIT

        # Compatibility check
        self._incompatible_modules = []
        # traverse the structure of Moduls
        pairs = (
            (self._spec_llm, [self._spec_solution]),
            (self._spec_solution, [self._spec_evaluator]),
            (self._spec_solution, self._spec_test),
            (self._spec_solution, self._spec_analysis),
            (self._spec_evaluator, self._spec_stat)
        )
        for (parent, children) in pairs:
            tags_out = parent.get_tags()['output']
            if len(tags_out) == 0:
                continue
            for child in children:
                tags_in = child.get_tags()['input']
                if len(tags_in) == 0:
                    continue
                if len(tags_out & tags_in) < 1:
                    self._incompatible_modules.append([parent.get_short_name(), child.get_short_name()])
                    self._state = TaskState.BREAK
            pass

        return True
