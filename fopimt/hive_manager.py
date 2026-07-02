import logging
import os
import random
import threading
import time
from datetime import UTC, datetime
from multiprocessing import Manager
from queue import Empty

from .hive import Hive, TaskPort
from .hive_dto import (
    RECIPIENT_BROADCAST,
    RECIPIENT_HIVE,
    SENDER_USER,
    HiveConfig,
    HiveInfo,
    HiveMessage,
    HiveMessageType,
    HivePolicyType,
    HiveState,
)
from .task_dto import OptimizationGoal, TaskState

# Task states considered terminal for hive completion detection
_TERMINAL_TASK_STATES = (TaskState.FINISH, TaskState.BREAK, TaskState.STOP)


class HiveInitializationException(Exception):
    def __init__(self, messages: list[str]):
        self.messages = messages


class HiveManager:
    """
    Owns the pool of Hives and runs the orchestrator monitor thread that
    routes messages between member Tasks (according to the Hive policy)
    and evaluates hive-level global stopping conditions.

    Member Tasks are executed by the existing TaskManager - the HiveManager
    only attaches TaskPorts before run and delegates run/pause/stop to Magic.
    """

    def __init__(self, magic):
        """
        :param magic: Magic instance (not typed to avoid circular import)
        """
        self._magic = magic
        self._hives: dict[str, Hive] = {}
        self._lock = threading.RLock()
        self._mp_manager = None  # lazy multiprocessing.Manager
        self._monitor_started = False

    ####################################################################
    #########  Public functions
    ####################################################################
    def load_from_disk(self) -> None:
        _dir = "out_task"
        if not os.path.exists(_dir):
            return
        hive_folders = [
            item
            for item in os.listdir(_dir)
            if os.path.isdir(os.path.join(_dir, item))
            and item.startswith(Hive.DIR_PREFIX)
        ]
        logging.info(f"Found {len(hive_folders)} hives.")
        for folder in hive_folders:
            try:
                hive: Hive = Hive.pickle_rick(folder)
                # A hive that was running when the server went down is broken
                if hive.state in (HiveState.RUN, HiveState.PAUSED):
                    hive.state = HiveState.BREAK
                    hive.log_error.append(
                        "Server restarted while the Hive was running."
                    )
                    hive.pickle_me()
                self._hives[hive.id] = hive
            except Exception as e:
                logging.error(
                    f"Unable to create instance of Hive from storage:{folder}. {e}"
                )

    def start_monitor(self) -> None:
        if self._monitor_started:
            return
        self._monitor_started = True
        threading.Thread(target=self._monitor, daemon=True).start()

    def hive_create(self, uid: str | None = None) -> Hive | None:
        with self._lock:
            if uid is not None and uid in self._hives:
                logging.warning(f"Hive with ID:{uid} already exists.")
                return self._hives[uid]
            hive = Hive.create_empty(hive_id=uid)
            if hive is None:
                return None
            self._hives[hive.id] = hive
            return hive

    def hive_get(self, uid: str) -> Hive | None:
        if uid in self._hives:
            return self._hives[uid]
        logging.error(f"Hive with ID:{uid} does not exist.")
        return None

    def hive_get_all(self) -> list[Hive]:
        return list(self._hives.values())

    def hive_init(self, uid: str, config: HiveConfig) -> Hive:
        """
        Initializes the Hive with the given config. Validates member Tasks
        and policy parameters. Raises HiveInitializationException with
        a list of human-readable messages on validation failure.
        """
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None:
                raise HiveInitializationException(
                    [f"Hive with the id {uid} does not exist."]
                )
            if hive.state in (HiveState.RUN, HiveState.PAUSED):
                raise HiveInitializationException(
                    ["Hive cannot be re-initialized while running."]
                )

            errors = self._validate_config(uid, config)
            if errors:
                raise HiveInitializationException(errors)

            # Detach tasks removed from the member list on re-init
            removed = set(hive.member_ids) - set(config.member_task_ids)
            for mid in removed:
                task = self._magic.task_get(mid)
                if task is not None:
                    task.hive_id = None
                    task.pickle_me()

            hive.initialize(config)

            # Mark membership on the Tasks (visible in TaskInfo.hive_id)
            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                task.hive_id = hive.id
                task.pickle_me()

            return hive

    def hive_run(self, uid: str) -> bool:
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None:
                logging.error(f"HiveManager:Run: Hive with id({uid}) does not exist.")
                return False

            # Resume of a paused hive
            if hive.state == HiveState.PAUSED:
                for mid in hive.member_ids:
                    task = self._magic.task_get(mid)
                    if task is not None and task.get_state() == TaskState.PAUSED:
                        self._magic.task_run(mid)
                hive.state = HiveState.RUN
                hive.pickle_me()
                return True

            if hive.state != HiveState.INIT:
                logging.error(
                    f"HiveManager:Run: Hive with id({uid}) is not initialized."
                )
                return False

            if os.name != "posix":
                logging.warning(
                    "Hive messaging requires Unix (fork). On this platform member "
                    "Tasks will run, but without cooperation."
                )

            # Validate all members are ready before starting any of them
            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                if task is None or task.get_state() != TaskState.INIT:
                    logging.error(
                        f"HiveManager:Run: Member Task ({mid}) of Hive ({uid}) "
                        f"is not in the INIT state."
                    )
                    return False

            # Build the communication runtime and attach ports
            manager = self._get_mp_manager()
            outbox = manager.Queue()
            inboxes = {}
            for mid in hive.member_ids:
                inboxes[mid] = manager.Queue()
            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                task.attach_port(
                    TaskPort(
                        task_id=mid,
                        hive_id=hive.id,
                        inbox=inboxes[mid],
                        outbox=outbox,
                    )
                )
            hive.attach_runtime(outbox, inboxes)
            hive.mark_run_start()
            hive.state = HiveState.RUN
            hive.pickle_me()

            for mid in hive.member_ids:
                if not self._magic.task_run(mid):
                    logging.error(
                        f"HiveManager:Run: Member Task ({mid}) of Hive ({uid}) "
                        f"could not be run."
                    )
                    hive.log_error.append(f"Member Task ({mid}) could not be run.")

            return True

    def hive_pause(self, uid: str) -> bool:
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None or hive.state != HiveState.RUN:
                logging.error(
                    f"HiveManager:Pause: Hive with id({uid}) is not in the RUN state."
                )
                return False

            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                if task is not None and task.get_state() == TaskState.RUN:
                    if not self._magic.task_pause(mid):
                        logging.error(
                            f"HiveManager:Pause: Member Task ({mid}) could not "
                            f"be paused."
                        )
            hive.state = HiveState.PAUSED
            hive.pickle_me()
            return True

    def hive_stop(self, uid: str, reason: str | None = None) -> bool:
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None or hive.state not in (HiveState.RUN, HiveState.PAUSED):
                logging.error(
                    f"HiveManager:Stop: Hive with id({uid}) is not in the RUN state."
                )
                return False

            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                if task is not None and task.get_state() in (
                    TaskState.RUN,
                    TaskState.PAUSED,
                ):
                    if not self._magic.task_stop(mid):
                        logging.error(
                            f"HiveManager:Stop: Member Task ({mid}) could not "
                            f"be stopped."
                        )
            hive.stop_reason = reason
            hive.state = HiveState.STOP
            hive.pickle_me()
            logging.info(f"Hive [{uid}] stopped. Reason: {reason or 'user request'}")
            return True

    def hive_archive(self, uid: str) -> bool:
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None:
                return False
            if not hive.archive():
                return False
            # Detach members
            for mid in hive.member_ids:
                task = self._magic.task_get(mid)
                if task is not None:
                    task.hive_id = None
                    task.pickle_me()
            self._hives.pop(uid, None)
            return True

    def inject_message(self, uid: str, recipient: str, payload: str) -> bool:
        """
        Manual (user/API) message injection into a running Hive.
        """
        with self._lock:
            hive = self._hives.get(uid)
            if hive is None:
                return False
            if hive.state not in (HiveState.RUN, HiveState.PAUSED):
                logging.error(
                    f"HiveManager:Message: Hive ({uid}) is not running - "
                    f"message cannot be delivered."
                )
                return False
            if recipient != RECIPIENT_BROADCAST and recipient not in hive.member_ids:
                logging.error(
                    f"HiveManager:Message: Recipient ({recipient}) is not "
                    f"a member of Hive ({uid})."
                )
                return False
            if hive.outbox is None:
                return False

        msg = HiveMessage(
            sender_id=SENDER_USER,
            recipient=recipient,
            msg_type=HiveMessageType.INFO,
            payload=payload,
        )
        hive.outbox.put(msg)
        return True

    def get_hive_info(self, uid: str) -> HiveInfo | None:
        hive = self._hives.get(uid)
        if hive is None:
            return None
        return hive.get_info(member_states=self._member_states(hive))

    def get_hives_info(self) -> list[HiveInfo]:
        return [
            hive.get_info(member_states=self._member_states(hive))
            for hive in self._hives.values()
        ]

    ####################################################################
    #########  Private functions
    ####################################################################
    def _get_mp_manager(self):
        if self._mp_manager is None:
            self._mp_manager = Manager()
        return self._mp_manager

    def _member_states(self, hive: Hive) -> dict[str, TaskState]:
        states = {}
        for mid in hive.member_ids:
            task = self._magic.task_get(mid)
            states[mid] = task.get_state() if task is not None else TaskState.STOP
        return states

    def _validate_config(self, uid: str, config: HiveConfig) -> list[str]:
        errors = []

        if len(config.member_task_ids) < 1:
            errors.append("Hive must have at least one member Task.")
        if len(set(config.member_task_ids)) != len(config.member_task_ids):
            errors.append("Duplicate Task ids in the member list.")

        for mid in config.member_task_ids:
            task = self._magic.task_get(mid)
            if task is None:
                errors.append(f"Member Task with the id {mid} does not exist.")
                continue
            if task.get_state() != TaskState.INIT:
                errors.append(
                    f"Member Task ({mid}) is not fully initialized "
                    f"(state: {task.get_state().name})."
                )
            member_of = getattr(task, "hive_id", None)
            if member_of is not None and member_of != uid:
                other = self._hives.get(member_of)
                if other is not None and other.state in (
                    HiveState.INIT,
                    HiveState.RUN,
                    HiveState.PAUSED,
                ):
                    errors.append(
                        f"Member Task ({mid}) already belongs to an active "
                        f"Hive ({member_of})."
                    )

        if config.policy == HivePolicyType.RANDOM_POOL:
            pool_size = config.policy_params.get("pool_size")
            if not isinstance(pool_size, int) or pool_size < 1:
                errors.append(
                    "Policy RANDOM_POOL requires an integer parameter "
                    "'pool_size' >= 1 in policy_params."
                )

        return errors

    def _monitor(self):
        """Monitor loop: routes bus messages and checks global stopping."""
        while True:
            try:
                with self._lock:
                    active = [
                        h
                        for h in self._hives.values()
                        if h.state in (HiveState.RUN, HiveState.PAUSED)
                    ]
                for hive in active:
                    self._process_hive(hive)
            except Exception as e:
                logging.error(f"HiveManager:Monitor: Unexpected error: {repr(e)}")
            time.sleep(0.5)

    def _process_hive(self, hive: Hive) -> None:
        # 1) Route pending messages from the shared outbox
        if hive.outbox is not None:
            while True:
                try:
                    msg: HiveMessage = hive.outbox.get_nowait()
                except Empty:
                    break
                except Exception as e:
                    logging.error(f"Hive[{hive.id}]: outbox read failed: {e}")
                    break
                hive.log_message(msg)
                self._route(hive, msg)

        if hive.state != HiveState.RUN:
            return

        # 2) Global (hive-level) stopping conditions
        reason = self._check_global_stopping(hive)
        if reason is not None:
            self.hive_stop(hive.id, reason=reason)
            return

        # 3) Completion detection - all members in a terminal state
        states = self._member_states(hive)
        if all(state in _TERMINAL_TASK_STATES for state in states.values()):
            if any(state == TaskState.BREAK for state in states.values()):
                hive.state = HiveState.BREAK
                hive.log_error.append("One or more member Tasks broke during run.")
            elif all(state == TaskState.FINISH for state in states.values()):
                hive.state = HiveState.FINISH
            else:
                hive.state = HiveState.STOP
            hive.pickle_me()
            logging.info(f"Hive [{hive.id}] completed with state: {hive.state.name}")

    def _route(self, hive: Hive, msg: HiveMessage) -> None:
        if msg.recipient == RECIPIENT_HIVE:
            self._handle_hive_addressed(hive, msg)
        elif msg.recipient == RECIPIENT_BROADCAST:
            targets = [m for m in hive.member_ids if m != msg.sender_id]
            self._deliver(hive, msg, targets)
        else:
            self._deliver(hive, msg, [msg.recipient])

    def _deliver(self, hive: Hive, msg: HiveMessage, targets: list[str]) -> None:
        for target in targets:
            inbox = hive.inboxes.get(target)
            if inbox is None:
                logging.warning(
                    f"Hive[{hive.id}]: no inbox for recipient ({target}), "
                    f"message dropped."
                )
                continue
            try:
                inbox.put(msg)
                hive.message_routed()
            except Exception as e:
                logging.error(f"Hive[{hive.id}]: delivery to ({target}) failed: {e}")

    def _handle_hive_addressed(self, hive: Hive, msg: HiveMessage) -> None:
        if msg.msg_type == HiveMessageType.ITERATION_REPORT:
            self._update_progress(hive, msg)
            self._apply_policy(hive, msg)
        else:
            # REQUEST / SPAWN / DESPAWN are phase 2 - only logged for now
            logging.info(
                f"Hive[{hive.id}]: received {msg.msg_type.name} from "
                f"({msg.sender_id}) - not supported yet, logged only."
            )

    def _update_progress(self, hive: Hive, report: HiveMessage) -> None:
        meta = report.metadata or {}
        hive.member_progress[report.sender_id] = {
            "iteration": int(meta.get("iteration", 0)),
            "used_tokens": int(meta.get("used_tokens", 0)),
        }

    def _apply_policy(self, hive: Hive, report: HiveMessage) -> None:
        if hive.policy == HivePolicyType.MANUAL:
            return

        meta = report.metadata or {}
        # Share only valid (state == 'OK') iteration results
        if meta.get("state") != "OK":
            return

        others = [m for m in hive.member_ids if m != report.sender_id]
        if not others:
            return

        targets = []
        match hive.policy:
            case HivePolicyType.BROADCAST_BEST:
                fitness = meta.get("fitness")
                if fitness is None:
                    return
                goal = (
                    hive.config.optimization_goal
                    if hive.config is not None
                    else OptimizationGoal.MINIMIZATION
                )
                best = hive.best_fitness
                if best is not None:
                    if goal == OptimizationGoal.MAXIMIZATION and fitness <= best:
                        return
                    if goal != OptimizationGoal.MAXIMIZATION and fitness >= best:
                        return
                hive.best_fitness = fitness
                targets = others
            case HivePolicyType.RING:
                try:
                    idx = hive.member_ids.index(report.sender_id)
                except ValueError:
                    return
                nxt = hive.member_ids[(idx + 1) % len(hive.member_ids)]
                if nxt == report.sender_id:
                    return
                targets = [nxt]
            case HivePolicyType.RANDOM:
                targets = [random.choice(others)]
            case HivePolicyType.RANDOM_POOL:
                pool_size = int(hive.config.policy_params.get("pool_size", 1))
                targets = random.sample(others, min(pool_size, len(others)))
            case _:
                return

        forward = HiveMessage(
            sender_id=report.sender_id,
            recipient=RECIPIENT_BROADCAST if len(targets) > 1 else targets[0],
            msg_type=HiveMessageType.INFO,
            payload=report.payload,
            metadata=meta,
        )
        self._deliver(hive, forward, targets)

    def _check_global_stopping(self, hive: Hive) -> str | None:
        cfg = hive.config.stopping if hive.config is not None else None
        if cfg is None:
            return None

        if cfg.max_total_iterations is not None:
            total = sum(p.get("iteration", 0) for p in hive.member_progress.values())
            if total >= cfg.max_total_iterations:
                return (
                    f"Global stopping condition met: total iterations "
                    f"({total}) >= max_total_iterations "
                    f"({cfg.max_total_iterations})."
                )

        if cfg.max_total_tokens is not None:
            total = sum(p.get("used_tokens", 0) for p in hive.member_progress.values())
            if total >= cfg.max_total_tokens:
                return (
                    f"Global stopping condition met: total used tokens "
                    f"({total}) >= max_total_tokens ({cfg.max_total_tokens})."
                )

        if cfg.max_time_seconds is not None and hive.time_run_start is not None:
            elapsed = (datetime.now(UTC) - hive.time_run_start).total_seconds()
            if elapsed >= cfg.max_time_seconds:
                return (
                    f"Global stopping condition met: elapsed time "
                    f"({int(elapsed)} s) >= max_time_seconds "
                    f"({cfg.max_time_seconds} s)."
                )

        return None
