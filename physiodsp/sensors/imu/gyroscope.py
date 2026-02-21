from numpy import array

from physiodsp.sensors.imu.base import IMUData


class GyroscopeData(IMUData):
    x: array
    y: array
    z: array
    fs: int = 64
