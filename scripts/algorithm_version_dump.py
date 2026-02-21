from physiodsp.activity.time_above_thr import TimeAboveThr
from physiodsp.activity.enmo import ENMO
from physiodsp.activity.zero_crossing import ZeroCrossing
from physiodsp.activity.pim import PIMAlgorithm
from physiodsp.ecg.peak_detector import EcgPeakDetector
from physiodsp.hrv.hrv_score import HrvScore


def main():

    print("| **Name** | **Version** |")
    print("|---|---|")
    print(f"| {TimeAboveThr().algorithm_name} | {TimeAboveThr().version} |")
    print(f"| {ENMO().algorithm_name} | {ENMO().version} |")
    print(f"| {ZeroCrossing().algorithm_name} | {ZeroCrossing().version} |")
    print(f"| {PIMAlgorithm().algorithm_name} | {PIMAlgorithm().version} |")
    print(f"| {EcgPeakDetector().algorithm_name} | {EcgPeakDetector().version} |")
    print(f"| {HrvScore().algorithm_name} | {HrvScore().version} |")


if __name__ == "__main__":
    main()
