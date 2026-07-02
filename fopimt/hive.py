import csv
import logging
import os
import pickle
import uuid
from datetime import UTC, datetime
from queue import Empty

from .hive_dto import HiveConfig, HiveInfo, HiveMessage, HivePolicyType, HiveState
from .magic_datetime import DateTime
from .task_dto import TaskState


def render_hive_message(msg: HiveMessage) -> str:
    """
    Renders a HiveMessage into a text block that is injected into the
    next LLM message of the receiving Task (via its buffer_message queue).
    """
    sender = None
    fitness = None
    if msg.metadata is not None:
        sender = msg.metadata.get("task_name")
        fitness = msg.metadata.get("fitness")
    sender = sender or msg.sender_id

    header = f"[Hive] Message from '{sender}'"
    if fitness is not None:
        header += f" (solution fitness: {fitness})"
    return f"{header}:\n{msg.payload}"


class TaskPort:
    """
    Communication port attached to a Task that is a member of a Hive.
    Holds an inbox (messages for the Task) and an outbox (shared queue
    towards the Hive orchestrator). Both queues are multiprocessing.Manager
    proxies, so they survive the fork of the Task process on Unix.

    The port is transient - it is never pickled with the Task
    (see Task.__getstate__) and is re-attached by the HiveManager
    on every hive run.
    """

    def __init__(self, task_id: str, hive_id: str, inbox, outbox):
        self._task_id = task_id
        self._hive_id = hive_id
        self._inbox = inbox
        self._outbox = outbox

    def get_task_id(self) -> str:
        return self._task_id

    def get_hive_id(self) -> str:
        return self._hive_id

    def pump(self) -> list[HiveMessage]:
        """
        Non-blocking drain of the inbox. Returns all pending messages.
        """
        msgs = []
        while True:
            try:
                msgs.append(self._inbox.get_nowait())
            except Empty:
                break
            except Exception as e:
                logging.error(f"TaskPort[{self._task_id}]: pump failed: {e}")
                break
        return msgs

    def publish(self, msg: HiveMessage) -> None:
        """
        Sends a message to the Hive orchestrator (never raises).
        """
        try:
            self._outbox.put(msg)
        except Exception as e:
            logging.error(f"TaskPort[{self._task_id}]: publish failed: {e}")


class Hive:
    """
    Orchestration unit grouping cooperating Tasks. Members remain regular
    Tasks in the Magic pool - the Hive only holds their ids, the routing
    policy and hive-level (global) stopping conditions. Message routing
    itself is executed by the HiveManager monitor thread.
    """

    DIR_PREFIX = "h_"
    DIR_ARCHIVE_PREFIX = "ah_"

    @classmethod
    def pickle_rick(cls, hive_folder: str):
        _file_name = os.path.join("out_task", hive_folder, "hive.pkl")
        with open(_file_name, "rb") as _file:
            return pickle.load(_file)

    @classmethod
    def create_empty(cls, hive_id: str | None = None):
        """Creates a new empty - uninitialized Hive object."""
        hive = cls()
        if hive_id is not None:
            hive._id = hive_id
        if not hive.create_dir():
            return None
        hive.pickle_me()
        return hive

    def __init__(self):
        # Descriptors
        self._id: str = str(uuid.uuid4())
        self._name: str = ""
        self._state: HiveState = HiveState.CREATED
        self._date: DateTime = DateTime()

        # Configuration
        self._config: HiveConfig | None = None
        self._member_ids: list[str] = []

        # Orchestration bookkeeping
        self._messages_routed: int = 0
        self._stop_reason: str | None = None
        self._log_error: list[str] = []
        self._best_fitness: float | None = None  # for BROADCAST_BEST policy
        # Last known progress per member (from ITERATION_REPORTs):
        # {task_id: {'iteration': int, 'used_tokens': int}}
        self._member_progress: dict[str, dict] = {}
        self._time_run_start: datetime | None = None

        # Storage
        self._dir: str = os.path.join("out_task")

        # Transient runtime communication (never pickled)
        self._outbox = None
        self._inboxes: dict[str, object] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # Manager queue proxies are not picklable / not valid across restarts
        state["_outbox"] = None
        state["_inboxes"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._outbox = None
        self._inboxes = {}

    ####################################################################
    #########  Public functions
    ####################################################################
    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> HiveState:
        return self._state

    @state.setter
    def state(self, value: HiveState):
        self._state = value

    @property
    def config(self) -> HiveConfig | None:
        return self._config

    @property
    def member_ids(self) -> list[str]:
        return self._member_ids

    @property
    def policy(self) -> HivePolicyType:
        if self._config is None:
            return HivePolicyType.MANUAL
        return self._config.policy

    @property
    def messages_routed(self) -> int:
        return self._messages_routed

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @stop_reason.setter
    def stop_reason(self, value: str | None):
        self._stop_reason = value

    @property
    def best_fitness(self) -> float | None:
        return self._best_fitness

    @best_fitness.setter
    def best_fitness(self, value: float | None):
        self._best_fitness = value

    @property
    def member_progress(self) -> dict[str, dict]:
        return self._member_progress

    @property
    def time_run_start(self) -> datetime | None:
        return self._time_run_start

    @property
    def outbox(self):
        return self._outbox

    @property
    def inboxes(self) -> dict[str, object]:
        return self._inboxes

    @property
    def log_error(self) -> list[str]:
        return self._log_error

    def initialize(self, config: HiveConfig) -> None:
        """
        Initializes the Hive from a HiveConfig. Member existence and state
        validation is the responsibility of the HiveManager (it has access
        to the Task pool). Moves state to INIT.
        """
        if config.name is not None:
            self._name = config.name
        self._config = config
        self._member_ids = list(config.member_task_ids)
        self._state = HiveState.INIT
        self._date.update_last_used()
        self.pickle_me()

    def attach_runtime(self, outbox, inboxes: dict[str, object]) -> None:
        """
        Attaches transient communication queues (Manager proxies) before run.
        """
        self._outbox = outbox
        self._inboxes = inboxes

    def mark_run_start(self) -> None:
        self._time_run_start = datetime.now(UTC)
        self._stop_reason = None
        self._member_progress = {}
        self._best_fitness = None

    def message_routed(self) -> None:
        self._messages_routed += 1

    def get_info(self, member_states: dict[str, TaskState] | None = None) -> HiveInfo:
        return HiveInfo(
            id=self._id,
            name=self._name,
            state=self._state,
            policy=self.policy,
            member_ids=self._member_ids,
            member_states=member_states or {},
            messages_routed=self._messages_routed,
            date_created=self._date.get_created_DateTime(),
            date_updated=self._date.get_created_DateTime(),
            stop_reason=self._stop_reason,
            log=self._log_error,
        )

    def create_dir(self) -> bool:
        try:
            self._dir = os.path.join(self._dir, self.DIR_PREFIX + self._id)
            os.mkdir(self._dir)
        except Exception as e:
            logging.error(
                f"Hive:{self._name}:{self._id} cant create folder! Exception: {e}"
            )
            return False
        return True

    def pickle_me(self) -> None:
        file_path = os.path.join(self._dir, "hive.pkl")
        try:
            with open(file_path, "wb") as file:
                pickle.dump(self, file)
        except Exception as e:
            logging.error(f"Error pickling Hive to {file_path}: {e}")

    def archive(self) -> bool:
        if self._state in [HiveState.RUN, HiveState.PAUSED]:
            return False
        old_dir = os.path.basename(self._dir)
        new_dir = (
            old_dir.replace(self.DIR_PREFIX, self.DIR_ARCHIVE_PREFIX, 1)
            if old_dir.startswith(self.DIR_PREFIX)
            else old_dir
        )
        os.rename(self._dir, os.path.join("out_task", new_dir))
        logging.info(f"Archived Hive:{self._name}:{self._id}")
        return True

    def log_message(self, msg: HiveMessage) -> None:
        """
        Appends a routed message to the bus log (audit/debug trail).
        """
        _path = os.path.join(self._dir, "bus_log.csv")
        try:
            _init = os.path.isfile(_path)
            with open(_path, "a", encoding="utf-8", newline="") as csv_file:
                fieldnames = [
                    "time",
                    "msg_id",
                    "sender",
                    "recipient",
                    "type",
                    "payload",
                    "metadata",
                ]
                wr = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
                if not _init:
                    wr.writeheader()
                wr.writerow(
                    {
                        "time": msg.timestamp,
                        "msg_id": msg.msg_id,
                        "sender": msg.sender_id,
                        "recipient": msg.recipient,
                        "type": msg.msg_type.name,
                        "payload": msg.payload,
                        "metadata": msg.metadata,
                    }
                )
        except Exception as e:
            logging.error(f"Hive[{self._id}]: failed to write bus log: {e}")
