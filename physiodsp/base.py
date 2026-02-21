from abc import ABC
from pandas import DataFrame
from numpy import ndarray


class BaseAlgorithm(ABC):

    # Class Attributes
    _algorithm_name = 'BaseAlgorithm'
    _version = 'v0.1.0'
    _window_len = 1
    _aggregation_window = 60

    def __init__(self) -> None:
        return None

    @property
    def algorithm_name(self) -> str:
        """Algorithm Name"""
        return self._algorithm_name

    @property
    def version(self) -> str:
        """Algorithm Version"""
        return self._version

    @property
    def window_len(self) -> int:
        """Window lenght in seconds"""
        return self._window_len

    @property
    def aggregation_window(self) -> int:
        """Aggregation Window in seconds"""
        return self._aggregation_window

    @classmethod
    def preprocess(self):
        raise NotImplementedError

    @classmethod
    def run(self):
        raise NotImplementedError

    @classmethod
    def aggregate(self,
                  timestamps: ndarray,
                  values: ndarray,
                  method: str = 'mean'
                  ):

        df = DataFrame(
            list(zip(timestamps, values)),
            columns=['timestamps', 'values']
        )

        df['timestamps'] = df[
            'timestamps'].apply(lambda x: (x // self._aggregation_window) * self._aggregation_window)

        if method == 'mean':
            df_agg = df.groupby('timestamps')[
                'values'].mean().reset_index(drop=False)

        self.biomarker_agg = df_agg
        return self
