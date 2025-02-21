import logging
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import current_process, set_start_method, Manager
import threading
from queue import Queue
from .task import Task, TaskState
import logging


# Simulated Task function (outside of class context)
def run_task(task: Task, task_pids=None, task_update_queue=None) -> Task:
    logging.info(f"Starting task: [{task.id}] (PID: {current_process().pid})")

    if task_pids != None:
        task_pids[task.get_id()] = current_process().pid

    stop_event = threading.Event()

    def update_task_state():
        while task.get_state() == TaskState.RUN and not stop_event.is_set():
            task_update_queue.put(task)
            time.sleep(0.5)  # Adjust the sleep interval as needed

    if task_update_queue:
        logging.info(f"Starting task update thread for: [{task.id}]")
        update_thread = threading.Thread(target=update_task_state)
        update_thread.start()

    def handle_stop(signum, frame):
        logging.info(f"Task [{task.id}] received signal: {signum}")
        stop_event.set() # Set the stop event to stop the update thread
        update_thread.join()  # Ensure the update thread finishes
        exit(0)

    signal.signal(signal.SIGTERM, handle_stop)

    try:
        task.run()
    except Exception as e:
        logging.error(f"Task [{task.id}] failed with exception: {e}")

    if task_update_queue:
        logging.info(f"Joining task update thread for: [{task.id}]")
        update_thread.join()  # Ensure the update thread finishes
    
    if task_pids != None:
        del task_pids[task.get_id()]
    return task


# TaskManager class with pause functionality
class TaskManager:
    def __init__(self, max_workers=4):
        self.running_on_unix = os.name == 'posix'

        # Live task updates require forking on Unix systems
        if self.running_on_unix:
            set_start_method('fork', force=True)
            self.manager = Manager()
            self.task_update_queue = self.manager.Queue()
            self.task_pids = self.manager.dict()

        self.executor = ProcessPoolExecutor(max_workers=max_workers)
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
                    if len(self.running_tasks) < self.executor._max_workers:
                        if not self.tasks_queue.empty():
                            task, callback = self.tasks_queue.get()

                            # 4 hours wasted on this shit. Needs linux to run properly.
                            if self.running_on_unix:
                                future = self.executor.submit(run_task, task, self.task_pids, self.task_update_queue)
                            else:
                                future = self.executor.submit(run_task, task)
                            self.running_tasks[task.id] = (future, callback)

                            # Attach a callback to handle task completion
                            future.add_done_callback(lambda f, n=task.id, cb=callback: self._task_completed(n, cb))

                # Live updates do not work on windows
                if self.running_on_unix:
                    with self.lock:
                        self._update_tasks_from_queue()                        

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
        updated_task = self.running_tasks[task_id][0].result()  # Get the updated Task object
        with self.lock:
            self._tasks[task_id] = updated_task  # Update the original task with its completed state
            del self.running_tasks[task_id]  # Clean up

        if callback:
            callback(updated_task)  # Call the callback with the updated task

        logging.info(f"Task [{task_id}] completed with state: {updated_task._state}")

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
                pid = self.task_pids[task_id]
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
                pid = self.task_pids[task_id]
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
                pid = self.task_pids[task_id]
                os.kill(pid, signal.SIGTERM)  # Send signal to terminate
            except Exception as e:
                logging.error(f"Failed to stop task [{task_id}]: {e}")
                return False

            logging.info(f"Task [{task_id}] (PID {pid}) has been stopped.")
            return True
        

    def shutdown(self):
        """Gracefully shut down the TaskManager."""
        self.shutdown_flag = True
        self.executor.shutdown(wait=True)
        logging.info("TaskManager has shut down.")

