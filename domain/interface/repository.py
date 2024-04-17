from abc import ABCMeta, abstractmethod
from typing import List
from pyspark.sql import DataFrame
from domain.entity.entities import DataColumns


class SparkRepository(metaclass=ABCMeta):
    @abstractmethod
    def read_csv(self, path: str) -> DataFrame:
        pass

    @abstractmethod
    def write_csv(self, df: DataFrame, path: str) -> None:
        pass

    @abstractmethod
    def get_data_columns(self) -> DataColumns:
        pass