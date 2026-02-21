from numpy import array

from physiodsp.sensors.imu.base import IMUData


class AccelerometerData(IMUData):
    x: array
    y: array
    z: array
    fs: int = 64
