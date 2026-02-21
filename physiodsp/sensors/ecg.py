from dataclasses import dataclass

from numpy import array


@dataclass
class EcgData:
    timestamps: array
    values: array
    fs: int = 64
