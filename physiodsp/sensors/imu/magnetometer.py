from numpy import array

from physiodsp.sensors.imu.base import IMUData


class MagnetometerData(IMUData):
    x: array
    y: array
    z: array
    fs: int = 64
