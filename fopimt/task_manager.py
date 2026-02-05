import logging
import os
import signal
import threading
import time
from multiprocessing import Manager, Process, current_process, set_start_method
from queue import Queue

from .task import Task, TaskState


# Simulated Task function (outside of class context)
def run_task(
    task: Task, task_pids=None, task_update_queue=None, task_result_queue=None
) -> Task:
    logging.info(f"Starting task: [{task.id}] (PID: {current_process().pid})")

    if task_pids != None:
        task_pids[task.get_id()] = current_process().pid

    stop_event = threading.Event()
    update_thread = None
    final_result_sent = False

    def update_task_state():
        while task.get_state() == TaskState.RUN and not stop_event.is_set():
            task_update_queue.put(task)
            time.sleep(0.5)  # Adjust the sleep interval as needed

    if task_update_queue:
        logging.info(f"Starting task update thread for: [{task.id}]")
        update_thread = threading.Thread(target=update_task_state)
        update_thread.start()

    def _set_task_state(state: TaskState):
        try:
            if hasattr(task, "set_state"):
                task.set_state(state)
            else:
                task._state = state
        except Exception as e:
            logging.error(f"Failed to set state for task [{task.id}]: {e}")

    def _send_final_result():
        nonlocal final_result_sent
        if final_result_sent:
            return
        if task_result_queue is not None:
            try:
                task_result_queue.put((task.get_id(), task))
                final_result_sent = True
            except Exception as e:
                logging.error(f"Task [{task.id}] failed to send result to queue: {e}")

    def handle_stop(signum, frame):
        logging.info(f"Task [{task.id}] received signal: {signum}")
        stop_event.set()  # Set the stop event to stop the update thread
        try:
            _set_task_state(TaskState.STOP)
            _send_final_result()
            if update_thread is not None:
                update_thread.join()  # Ensure the update thread finishes
        except Exception as e:
            logging.error(f"Task [{task.id}] failed to stop cleanly: {e}")
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, handle_stop)

    try:
        task.run()
    except Exception as e:
        logging.error(f"Task [{task.id}] failed with exception: {e}")
    finally:
        _send_final_result()

    if task_update_queue:
        logging.info(f"Joining task update thread for: [{task.id}]")
        update_thread.join()  # Ensure the update thread finishes

    if task_pids != None:
        del task_pids[task.get_id()]
    return task


# TaskManager class with pause functionality
class TaskManager:
    def __init__(self, max_workers=4):
        self.running_on_unix = os.name == "posix"
        self.max_workers = max_workers

        # Live task updates require forking on Unix systems
        if self.running_on_unix:
            set_start_method("fork", force=True)
            self.manager = Manager()
            self.task_update_queue = self.manager.Queue()
            self.task_result_queue = self.manager.Queue()
            self.task_pids = self.manager.dict()
        else:
            self.task_result_queue = Queue()

        self.tasks_queue = Queue()
        self.running_tasks = {}
        self.lock = threading.Lock()
        self._tasks = {}  # Dictionary to store Task instances by ID
        self.shutdown_flag = False

    def add_task(self, task: Task, callback=None):
        """Add a task to the queue with an optional callback for completion."""
        self.tasks_queue.put((task, callback))
        self._tasks[task.id] = task  # Store task reference
        logging.info(f"Task [{task.id}] added to the queue.")

    def start_monitor(self):
        """Monitor loop to handle task submission and completion."""

        def monitor():
            while not self.shutdown_flag or not self.tasks_queue.empty():
                with self.lock:
                    if len(self.running_tasks) < self.max_workers:
                        if not self.tasks_queue.empty():
                            task, callback = self.tasks_queue.get()

                            # 4 hours wasted on this shit. Needs linux to run properly.
                            logging.info(
                                f"Starting new task process from the queue. [{task.id}]"
                            )
                            if self.running_on_unix:
                                process = Process(
                                    target=run_task,
                                    args=(
                                        task,
                                        self.task_pids,
                                        self.task_update_queue,
                                        self.task_result_queue,
                                    ),
                                )
                            else:
                                process = Process(
                                    target=run_task,
                                    args=(task, None, None, self.task_result_queue),
                                )

                            process.start()
                            if self.running_on_unix:
                                self.task_pids[task.id] = process.pid
                            self.running_tasks[task.id] = (process, callback)

                # Live updates do not work on windows
                if self.running_on_unix:
                    with self.lock:
                        self._update_tasks_from_queue()

                self._check_completed_processes()
                time.sleep(0.5)  # Polling interval for task management

        threading.Thread(target=monitor, daemon=True).start()

    def _update_tasks_from_queue(self):
        while not self.task_update_queue.empty():
            updated_task: Task = self.task_update_queue.get()
            running_task = self.running_tasks.get(updated_task.get_id())

            if running_task:
                _, callback = running_task
                self._tasks[updated_task.get_id()] = updated_task

                if callback:
                    callback(updated_task)

    def _task_completed(self, task_id: str, callback=None):
        """Handle task completion, update the dictionary, and invoke callback if provided."""
        updated_task = self._tasks.get(task_id)
        while not self.task_result_queue.empty():
            try:
                result_task_id, result_task = self.task_result_queue.get()
                self._tasks[result_task_id] = result_task
                if result_task_id == task_id:
                    updated_task = result_task
            except Exception as e:
                logging.error(f"Failed to read task result for [{task_id}]: {e}")
                break
        with self.lock:
            if updated_task is not None:
                self._tasks[task_id] = updated_task
            del self.running_tasks[task_id]
            if self.running_on_unix and task_id in self.task_pids:
                del self.task_pids[task_id]

        if callback:
            callback(updated_task)  # Call the callback with the updated task

        if updated_task is not None:
            logging.info(
                f"Task [{task_id}] completed with state: {updated_task._state}"
            )
        else:
            logging.info(f"Task [{task_id}] completed.")

    def _check_completed_processes(self):
        completed_tasks = []
        with self.lock:
            for task_id, (process, callback) in list(self.running_tasks.items()):
                if not process.is_alive():
                    process.join()
                    completed_tasks.append((task_id, callback))

        for task_id, callback in completed_tasks:
            self._task_completed(task_id, callback)

    def pause_task(self, task_id: str) -> bool:
        """Attempt to pause a running task."""
        # TODO: Check if the task is in a state that can be paused
        if not self.running_on_unix:
            logging.error("Pause functionality is only supported on Unix systems.")
            return False

        with self.lock:
            if task_id not in self.running_tasks:
                return False

            if task_id not in self.task_pids or not self.task_pids[task_id]:
                logging.error(f"Can't find PID of task [{task_id}].")
                return False

            try:
                # Make sure the task is updated before pausing
                self._update_tasks_from_queue()
                current_task = self._tasks.get(task_id)
                if current_task is not None:
                    if hasattr(current_task, "set_state"):
                        current_task.set_state(TaskState.PAUSED)
                    else:
                        current_task._state = TaskState.PAUSED
                    self._tasks[task_id] = current_task
                    _, callback = self.running_tasks[task_id]
                    if callback:
                        callback(current_task)
                process, _ = self.running_tasks[task_id]
                pid = self.task_pids.get(task_id) or process.pid
                os.kill(pid, signal.SIGSTOP)  # Send signal to pause
            except Exception as e:
                logging.error(f"Failed to pause task [{task_id}]: {e}")

            logging.info(f"Task [{task_id}] (PID {pid}) has been paused.")
            return True

    def resume_task(self, task_id: str):
        """Attempt to resume a paused task."""
        # TODO: Check if the task is paused before resuming
        if not self.running_on_unix:
            logging.error("Pause functionality is only supported on Unix systems.")
            return False

        with self.lock:
            if task_id not in self.running_tasks:
                return False

            if task_id not in self.task_pids or not self.task_pids[task_id]:
                logging.error(f"Can't find PID of task [{task_id}].")
                return False

            try:
                current_task = self._tasks.get(task_id)
                if current_task is not None:
                    if hasattr(current_task, "set_state"):
                        current_task.set_state(TaskState.RUN)
                    else:
                        current_task._state = TaskState.RUN
                    self._tasks[task_id] = current_task
                    _, callback = self.running_tasks[task_id]
                    if callback:
                        callback(current_task)
                process, _ = self.running_tasks[task_id]
                pid = self.task_pids.get(task_id) or process.pid
                os.kill(pid, signal.SIGCONT)  # Send signal to resume
            except Exception as e:
                logging.error(f"Failed to resume task [{task_id}]: {e}")

            logging.info(f"Task [{task_id}] (PID {pid}) has been resumed.")
            return True

    def stop_task(self, task_id: str) -> bool:
        """Forcefully stop a running task by terminating its process."""
        if not self.running_on_unix:
            logging.error("Stop functionality is only supported on Unix systems.")
            return False

        with self.lock:
            if task_id not in self.running_tasks:
                logging.error(f"Task [{task_id}] is not running.")
                return False

            if task_id not in self.task_pids or not self.task_pids[task_id]:
                logging.error(f"Can't find PID of task [{task_id}].")
                return False

            try:
                self._update_tasks_from_queue()
                process, _ = self.running_tasks[task_id]
                pid = self.task_pids.get(task_id) or process.pid
                os.kill(pid, signal.SIGTERM)  # Send signal to terminate
            except Exception as e:
                logging.error(f"Failed to stop task [{task_id}]: {e}")
                return False

            logging.info(f"Task [{task_id}] (PID {pid}) has been stopped.")
            return True

    def shutdown(self):
        """Gracefully shut down the TaskManager."""
        self.shutdown_flag = True
        with self.lock:
            for _, (process, _) in list(self.running_tasks.items()):
                if process.is_alive():
                    process.terminate()
                process.join()
            self.running_tasks.clear()
        logging.info("TaskManager has shut down.")
