from enum import Enum
from .cached import CachedSpectrumCollection
from .sharedarray import SharedArraySpectrumCollection


class SpectrumCollectionType(Enum):
    CACHED = CachedSpectrumCollection
    SHAREDARRAY = SharedArraySpectrumCollection
