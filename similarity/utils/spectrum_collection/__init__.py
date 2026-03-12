from enum import Enum
from .cached import CachedSpectrumCollection


class SpectrumCollectionType(Enum):
    CACHED = CachedSpectrumCollection
