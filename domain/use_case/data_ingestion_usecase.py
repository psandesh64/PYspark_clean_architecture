from domain.entity.entities import DataIngestionConfig
from data.spark_repository import InSparkRepository

class DataIngestionUseCase:
    def __init__(self, config: DataIngestionConfig, repo: InSparkRepository):
        self.config = config
        self.repo = repo

    def execute(self,csv_file_path):
        df = self.repo.read_csv(csv_file_path)
        self.repo.write_csv(df, self.config.raw_data_path)
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        self.repo.write_csv(train_df, self.config.train_data_path)
        self.repo.write_csv(test_df, self.config.test_data_path)
        return self.config.train_data_path, self.config.test_data_path