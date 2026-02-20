from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
import multiprocessing as mp
import diskcache
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from types import UnionType

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
    @staticmethod
    def get_type(ftype):
        if isinstance(ftype, UnionType):
            return ftype.__args__[0]
        return ftype

    @staticmethod
    def get_required(field):
        if isinstance(field.type, UnionType):
            return field.default is None and type(None) not in field.type.__args__
        return field.default is None

    @classmethod
    def argparser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Experiment configuration")
        for field in fields(cls):
            kw = dict(default=field.default, required=cls.get_required(field))
            if field.type is bool:
                # for bools, use action='store_true' and default to False
                kw["action"] = "store_true"
            else:
                kw["type"] = cls.get_type(field.type)
            parser.add_argument(f"--{field.name.replace('_', '-')}", **kw)
        return parser


@dataclass(frozen=True, slots=True)
class Config(BaseConfig):
    input_file: Path
    collision_energy: int = 30
    charge: int = 2
    model_intensity: str = "Prosit_2020_intensity_HCD"
    model_irt: str = "Prosit_2019_irt"
    model_ccs: str | None = None
    mz_tolerance: float = 1.0
    irt_tolerance: float = 5.0
    peak_tolerance: float = 0.0
    peak_ppm: float = 10.0
    ccs_rtolerance: float = 0.02
    nonstandard_aminoacids: bool = False
    koina_host: str = "koina.wilhelmlab.org:443"
    cache_dir: Path = Path(".")
    workers: int = mp.cpu_count()


class Index(diskcache.Index, ABC):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    @abstractmethod
    def _full_key(self, key: str) -> tuple:
        pass

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment
        super().__init__(str(experiment.config.cache_dir))

    def __getitem__(self, key: str) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __setitem__(self, key: str, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)

    def __contains__(self, key: str) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work


class SpectrumIndex(Index):
    def _full_key(self, key: str) -> tuple:
        config = self.experiment.config
        return (key, config.collision_energy, config.charge, config.model_intensity)


class RTIndex(Index):
    def _full_key(self, key: str) -> tuple:
        config = self.experiment.config
        return (key, config.model_irt)


class IMIndex(Index):
    def _full_key(self, key: str) -> tuple:
        config = self.experiment.config
        return (key, config.charge, config.model_ccs)


class ExperimentWorker(mp.Process):
    def __init__(
        self, task_queue: mp.Queue, result_queue: mp.Queue, experiment: "Experiment"
    ):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.experiment = experiment

    @abstractmethod
    def run(self) -> None:
        return super().run()
