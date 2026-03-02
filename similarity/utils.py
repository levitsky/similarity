from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
import multiprocessing as mp
import diskcache
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from types import UnionType
import logging

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class Fixture(ABC):
    """A descriptor for a calculation result. The result is calculated lazily on first access and then cached for subsequent accesses."""

    _data: dict["Experiment", Any] = {}

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Any:
        pass

    def __init__(self):
        super().__init__()
        self._data = {}

    def __get__(self, obj, objtype=None):
        logger.debug(
            "Accessing %s fixture %s on obj %s. Keys cached are currently:\n%s",
            self.__class__.__name__,
            self,
            obj,
            self._data.keys(),
        )
        if obj not in self._data:
            logger.debug("%s not in cache on %s, evaluating...", obj, self)
            self._data[obj] = self.evaluate(obj)
            logger.info(
                "Finished evaluating %s for %s %d",
                self.__class__.__name__,
                obj.__class__.__name__,
                id(obj),
            )
        else:
            logger.debug("%s already in cache: (type %s)", self, type(self._data[obj]))
        return self._data[obj]

    def __set__(self, obj, value):
        raise AttributeError(f"Cannot set value of fixture {self.__class__.__name__}")


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
    fragmentation_type: str | None = None
    min_charge: int = 2
    max_charge: int = 2
    model_intensity: str = "Prosit_2020_intensity_HCD"
    model_irt: str = "Prosit_2019_irt"
    model_ccs: str | None = None
    mz_tolerance: float = 1.0
    irt_tolerance: float = 5.0
    peak_tolerance: float = 0.0
    peak_ppm: float = 10.0
    ccs_rtolerance: float = 0.02
    nonstandard_aminoacids: bool = False
    ptms: bool = False
    koina_host: str = "koina.wilhelmlab.org:443"
    cache_dir: Path = Path(".")
    cache_properties: bool = False
    workers: int = mp.cpu_count()
    batch_size: int = 100000
    score_threshold: float = 0.0


class Index(diskcache.Index, ABC):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    @abstractmethod
    def _full_key(self, key: str) -> tuple:
        pass

    def __new__(cls, experiment: "Experiment", *args, **kwargs):
        assert (
            experiment._cache is not None
        ), "Experiment cache must be initialized before creating Index"
        instance = super().__new__(cls)
        instance._cache = experiment._cache
        return instance

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment
        # not calling super().__init__() because it would reassign self._cache

    def __getitem__(self, key: Any) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __setitem__(self, key: Any, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)

    def __contains__(self, key: Any) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work


class SpectrumIndex(Index):
    def _full_key(self, key: tuple[str, int]) -> bytes:
        config = self.experiment.config
        return bytes(
            f"{key[0]}_{config.collision_energy}_{key[1]}_{config.model_intensity}",
            "ascii",
        )


class RTIndex(Index):
    def _full_key(self, key: str) -> bytes:
        config = self.experiment.config
        return bytes(f"{key}_{config.model_irt}", "ascii")


class IMIndex(Index):
    def _full_key(self, key: tuple[str, int]) -> bytes:
        config = self.experiment.config
        return bytes(f"{key[0]}_{key[1]}_{config.model_ccs}", "ascii")


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
