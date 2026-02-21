from dataclasses import dataclass

from numpy import array, sqrt, column_stack


@dataclass
class IMUData:
    timestamps: array
    x: array
    y: array
    z: array
    fs: int = 64

    @property
    def magnitude(self) -> array:
        """Inertial Measurement Unit Magnitude"""
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_matrix(self) -> array:
        """
        Converte the three axes into a (N, 3) numpy ndarray.

        Returns:
            array: Array of shape (N, 3) with columns [x, y, z] where N is the number of samples
        """
        return column_stack((self.x, self.y, self.z))
