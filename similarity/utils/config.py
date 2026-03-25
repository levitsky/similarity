from pathlib import Path
import argparse
from dataclasses import dataclass, fields
from types import UnionType
import multiprocessing as mp
from enum import EnumType, Enum, auto
from .cache import CacheType
from .spectrum_collection import SpectrumCollectionType
from abc import ABC
from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from dataclasses import Field


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaseConfig(ABC):
    @staticmethod
    def get_type(ftype):
        if isinstance(ftype, UnionType):
            return ftype.__args__[0]
        if isinstance(ftype, EnumType):
            return None
        return ftype

    @staticmethod
    def get_default_enum(field):
        if isinstance(field.type, EnumType):
            return field.default.name
        return field.default

    @staticmethod
    def get_required(field):
        if isinstance(field.type, UnionType):
            return field.default is None and type(None) not in field.type.__args__
        return field.default is None

    @classmethod
    def _args_for_field(cls, field) -> tuple[str, dict]:
        name = f"--{field.name.replace('_', '-')}"
        kw: dict[str, Any] = dict(
            default=field.default, required=cls.get_required(field)
        )
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
            kw["default"] = cls.get_default_enum(field)
            kw["choices"] = list(choice.name for choice in field.type)
        return name, kw

    @classmethod
    def argparser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Experiment configuration")
        for field in fields(cls):
            if field.name == "cache_conf":
                cache_group = parser.add_argument_group("Cache configuration")

                for fn, subfs in cache_args.items():
                    c, subf = subfs[0]
                    name, kw = c.value._args_for_field(subf)
                    help_text = f"(used for {', '.join(cf.name for cf, _ in subfs)} cache configuration)"
                    kw["help"] = help_text
                    cache_group.add_argument(name, **kw)
            else:
                name, kw = cls._args_for_field(field)
                parser.add_argument(name, **kw)
        return parser


@dataclass(frozen=True, slots=True)
class CacheConfig(BaseConfig):
    cache_properties: bool = False


@dataclass(frozen=True, slots=True)
class DiskCacheConfig(CacheConfig):
    cache_dir: Path = Path.cwd()


@dataclass(frozen=True, slots=True)
class RedisCacheConfig(CacheConfig):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


class CacheConfigType(Enum):
    DISKCACHE = DiskCacheConfig
    # REDIS = RedisCacheConfig


cache_args: dict[str, list[tuple[CacheConfigType, "Field"]]] = {}
"""
Cache configuration arguments. Keys are names of fields, values are lists of tuples of (CacheConfigType, Field) for each cache configuration type that uses the field.
"""
for c in CacheConfigType:
    for subf in fields(c.value):
        cache_args.setdefault(subf.name, []).append((c, subf))


class LiteralEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class KoinaIntensityModel(LiteralEnum):
    AlphaPeptDeep_ms2_generic = auto()
    Altimeter_2024_intensities = auto()
    Altimeter_2024_isotopes = auto()
    Altimeter_2024_splines = auto()
    Altimeter_2024_splines_index = auto()
    Prosit_2019_intensity = auto()
    Prosit_2020_intensity_CID = auto()
    Prosit_2020_intensity_HCD = auto()
    Prosit_2020_intensity_TMT = auto()
    Prosit_2023_intensity_timsTOF = auto()
    Prosit_2024_intensity_PTMs_gl = auto()
    Prosit_2024_intensity_cit = auto()
    Prosit_2025_intensity_22PTM = auto()
    Prosit_2025_intensity_40PTM = auto()
    Prosit_2025_intensity_MultiFrag = auto()
    Prosit_2025_intensity_lac = auto()
    UniSpec = auto()
    ms2pip_CID_TMT = auto()
    ms2pip_HCD2021 = auto()
    ms2pip_Immuno_HCD = auto()
    ms2pip_TTOF5600 = auto()
    ms2pip_iTRAQphospho = auto()
    ms2pip_timsTOF2023 = auto()
    ms2pip_timsTOF2024 = auto()


class KoinaRTModel(LiteralEnum):
    AlphaPeptDeep_rt_generic = auto()
    Chronologer_RT = auto()
    Deeplc_hela_hf = auto()
    Prosit_2019_irt = auto()
    Prosit_2020_irt_TMT = auto()
    Prosit_2024_irt_PTMs_gl = auto()
    Prosit_2024_irt_cit = auto()
    Prosit_2025_irt_22PTM = auto()
    Prosit_2025_irt_40PTM = auto()
    Prosit_2025_irt_lac = auto()


class KoinaCCSModel(LiteralEnum):
    AlphaPeptDeep_ccs_generic = auto()
    IM2Deep = auto()


@dataclass(frozen=True, slots=True)
class Config(BaseConfig):
    input_file: Path
    collision_energy: int = 30
    fragmentation_type: str | None = None
    min_charge: int = 2
    max_charge: int = 2
    model_intensity: KoinaIntensityModel = KoinaIntensityModel.Prosit_2020_intensity_HCD
    model_irt: KoinaRTModel = KoinaRTModel.Prosit_2019_irt
    model_ccs: KoinaCCSModel | None = None
    mz_tolerance: float = 1.0
    irt_tolerance: float = 5.0
    peak_tolerance: float = 0.0
    peak_ppm: float = 10.0
    ccs_rtolerance: float = 0.02
    nonstandard_aminoacids: bool = False
    ptms: bool = False
    koina_host: str = "koina.wilhelmlab.org:443"
    cache: CacheType = CacheType.NONE
    cache_conf: CacheConfigType | None = None
    workers: int = mp.cpu_count()
    batch_size: int = 1000
    score_threshold: float = 0.0
    spectrum_collection: SpectrumCollectionType = SpectrumCollectionType.SHAREDARRAY
    max_peaks: int = 50

    def __post_init__(self):
        if self.cache != CacheType.NONE and self.cache_conf is None:
            logger.warning(
                "cache_conf should be provided when cache is enabled. Using default configuration for {self.cache.name}."
            )
            object.__setattr__(
                self, "cache_conf", CacheConfigType[self.cache.name].value()
            )
