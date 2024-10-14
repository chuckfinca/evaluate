from enum import Enum, auto

class TaskType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"

    def __init__(self, task_type):
        self._task_type = task_type

    @property
    def value(self):
        return self._task_type

    def __str__(self):
        return self.task_type.replace('_', ' ')