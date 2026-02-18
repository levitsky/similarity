from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
import multiprocessing as mp
import diskcache
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
        # key is peptide sequence, we need to add collision energy, charge and model info to the key
        return (key, config.collision_energy, config.charge, config.model_intensity)


class RTIndex(Index):
    def _full_key(self, key: str) -> tuple:
        config = self.experiment.config
        # key is peptide sequence, we need to add model info to the key
        return (key, config.model_irt)
