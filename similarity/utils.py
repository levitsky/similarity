from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
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


@dataclass(frozen=True, slots=True)
class BaseConfig:
    @classmethod
    def argparser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Experiment configuration")
        for field in fields(cls):
            parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                type=field.type,
                default=field.default,
                required=field.default is None,
            )
        return parser


@dataclass(frozen=True, slots=True)
class Config(BaseConfig):
    # workdir: Path
    input_file: Path
    collision_energy: int = 30
    charge: int = 2
    model_intensity: str = "Prosit_2020_intensity_HCD"
    model_irt: str = "Prosit_2019_irt"
    mz_tolerance: float = 1.0
    irt_tolerance: float = 5.0
    peak_tolerance: float = 0.0
    peak_ppm: float = 10.0
    koina_host: str = "koina.wilhelmlab.org:443"
