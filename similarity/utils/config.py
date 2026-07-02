from pathlib import Path
import argparse
from dataclasses import dataclass, fields
from types import UnionType
import multiprocessing as mp
from enum import EnumType, Enum, auto
from .cache import CacheType
from .spectrum_collection import SpectrumCollectionType
from abc import ABC
from typing import Any, TYPE_CHECKING, get_origin, get_args
import logging
from pyteomics.mass.mass import PROTON, nist_mass

if TYPE_CHECKING:
    from dataclasses import Field


logger = logging.getLogger(__name__)
PROTON_MASS = nist_mass[PROTON][0][0]


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
        elif isinstance(kw.get("type"), EnumType):  # assume it's an optional Enum
            kw["choices"] = list(choice.name for choice in kw["type"]) + [None]
            kw["type"] = (
                None  # argparse doesn't support optional Enums, so we have to handle None as a special case
            )
        elif get_origin(kw.get("type")) is list and get_args(kw.get("type")) == (str,):
            kw["type"] = str
            kw["nargs"] = "*" if field.default is None else "+"
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
    REDIS = RedisCacheConfig


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


class MzErrorUnit(LiteralEnum):
    Th = auto()
    ppm = auto()
    Da = "Th"  # alias for Th
    PPM = "ppm"  # alias for ppm


class FragmentationType(LiteralEnum):
    HCD = auto()
    CID = auto()


@dataclass(frozen=True, slots=True)
class Config(BaseConfig):
    collision_energy: int = 30
    fragmentation_type: FragmentationType = FragmentationType.HCD
    min_charge: int = 2
    max_charge: int = 2
    min_length: int = 5
    max_length: int = 30
    model_intensity: KoinaIntensityModel = (
        KoinaIntensityModel.Prosit_2025_intensity_40PTM
    )
    model_irt: KoinaRTModel = KoinaRTModel.Prosit_2025_irt_40PTM
    model_ccs: KoinaCCSModel | None = None
    precursor_mz_tolerance: float = 10.0
    precursor_mz_unit: MzErrorUnit = MzErrorUnit.PPM
    isotope_error: int = 1
    irt_tolerance: float = 5.0
    fragment_mz_tolerance: float = 10.0
    fragment_mz_unit: MzErrorUnit = MzErrorUnit.PPM
    ccs_rtolerance: float = 0.02
    nonstandard_aminoacids: bool = False
    ptms: bool = False
    fixed_mods: list[str] | None = None
    variable_mods: list[str] | None = None
    koina_host: str = "koina.wilhelmlab.org:443"
    cache: CacheType = CacheType.NONE
    cache_conf: CacheConfigType | None = None
    workers: int = mp.cpu_count()
    batch_size: int = 1000
    score_threshold: float = 0.0
    spectrum_collection: SpectrumCollectionType = SpectrumCollectionType.SHAREDARRAY
    max_peaks: int = 50
    subsets: int = 1
    subset: int = 0

    def absolute_mz_error(self, mz: float) -> float:
        if self.precursor_mz_unit == MzErrorUnit.PPM:
            return mz * self.precursor_mz_tolerance / 1e6
        elif self.precursor_mz_unit == MzErrorUnit.Th:
            return self.precursor_mz_tolerance
        else:
            raise ValueError(f"Unsupported m/z error unit: {self.precursor_mz_unit}")

    def within_mz_tolerance(self, mz1: float, mz2: float) -> bool:
        if mz1 > mz2:
            mz1, mz2 = mz2, mz1
        return mz2 - mz1 <= self.absolute_mz_error(mz2)

    def __post_init__(self):
        if self.cache != CacheType.NONE and self.cache_conf is None:
            logger.warning(
                f"cache_conf should be provided when cache is enabled. Using default configuration for {self.cache.name}."
            )
            object.__setattr__(
                self, "cache_conf", CacheConfigType[self.cache.name].value()
            )
