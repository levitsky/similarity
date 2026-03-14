from pathlib import Path
import argparse
from dataclasses import dataclass, fields
from types import UnionType
import multiprocessing as mp
from enum import EnumType
from .cache import CacheType
from .spectrum_collection import SpectrumCollectionType


@dataclass(frozen=True, slots=True)
class BaseConfig:
    @staticmethod
    def get_type(ftype):
        if isinstance(ftype, UnionType):
            return ftype.__args__[0]
        if isinstance(ftype, EnumType):
            return lambda name: getattr(ftype, name)
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
            name = f"--{field.name.replace('_', '-')}"
            kw = dict(default=field.default, required=cls.get_required(field))
            if field.type is bool:
                if field.default:
                    kw["action"] = "store_false"
                    kw["dest"] = field.name
                    name = f"--no-{field.name.replace('_', '-')}"
                else:
                    kw["action"] = "store_true"
            else:
                kw["type"] = cls.get_type(field.type)
            if isinstance(field.type, EnumType):
                kw["choices"] = list(field.type)
            parser.add_argument(name, **kw)
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
    cache: CacheType = CacheType.NONE
    cache_dir: Path = Path(".")
    cache_properties: bool = False
    workers: int = mp.cpu_count()
    batch_size: int = 5000
    score_threshold: float = 0.0
    spectrum_collection: SpectrumCollectionType = SpectrumCollectionType.SHAREDARRAY
    max_peaks: int = 50
