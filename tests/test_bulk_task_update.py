import unittest
from types import SimpleNamespace

from fopimt.magic import Magic
from fopimt.task import TaskInitializationException
from fopimt.task_dto import TaskState


class _FakeTask:
    def __init__(self, task_id: str, state: TaskState = TaskState.INIT):
        self.id = task_id
        self.state = state
        self.value = "original"
        self.persisted = False

    def get_state(self):
        return self.state

    def initialize(self, loader, task_config, persist=True):
        if task_config == "invalid":
            raise TaskInitializationException(["Invalid staged configuration."])
        self.value = task_config
        self.state = TaskState.INIT
        if persist:
            self.pickle_me()

    def pickle_me(self):
        self.persisted = True


def _update(task_id: str, configuration: str):
    return SimpleNamespace(
        task_id=task_id,
        task_configuration=configuration,
    )


class BulkTaskUpdateTests(unittest.TestCase):
    def create_manager(self):
        manager = Magic.__new__(Magic)
        manager._loader = object()
        manager._tasks = {
            "one": _FakeTask("one"),
            "two": _FakeTask("two"),
        }
        return manager

    def test_commits_all_candidates_after_validation(self):
        manager = self.create_manager()
        originals = manager._tasks.copy()

        updated = manager.task_update_batch(
            [_update("one", "first"), _update("two", "second")]
        )

        self.assertEqual(["first", "second"], [task.value for task in updated])
        self.assertTrue(all(task.persisted for task in updated))
        self.assertIsNot(originals["one"], manager._tasks["one"])
        self.assertIsNot(originals["two"], manager._tasks["two"])

    def test_validation_failure_keeps_every_original_task(self):
        manager = self.create_manager()
        originals = manager._tasks.copy()

        with self.assertRaises(TaskInitializationException):
            manager.task_update_batch(
                [_update("one", "first"), _update("two", "invalid")]
            )

        self.assertIs(originals["one"], manager._tasks["one"])
        self.assertIs(originals["two"], manager._tasks["two"])
        self.assertFalse(originals["one"].persisted)
        self.assertFalse(originals["two"].persisted)

    def test_rejects_non_editable_tasks(self):
        manager = self.create_manager()
        manager._tasks["one"].state = TaskState.RUN

        with self.assertRaises(TaskInitializationException):
            manager.task_update_batch([_update("one", "first")])


if __name__ == "__main__":
    unittest.main()
