import multiprocessing as mp
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ExperimentWorker(ABC, mp.Process):
    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, **kwargs):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self) -> None:
        return super().run()
