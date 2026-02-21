from dataclasses import dataclass

from numpy import array


@dataclass
class HrvData:
    timestamps: array
    values: array
