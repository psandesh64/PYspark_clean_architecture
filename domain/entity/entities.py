import os
from dataclasses import dataclass
from typing import List

class DataColumns:
    def __init__(self, numerical_columns: List[str], categorical_columns: List[str], target_column: str):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor_rf')

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('artifacts', 'model_rf')