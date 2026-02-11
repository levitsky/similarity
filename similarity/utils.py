from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import Experiment


class Fixture(ABC):
    """A descriptor for a calculation result. Placeholder for caching implementation."""
    _data: Any = None

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Any:
        pass

    def __get__(self, obj, objtype=None):
        if self._data is None:
            self._data = self.evaluate(obj)
        return self._data