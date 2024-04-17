from domain.interface.repository import SparkRepository
from domain.entity.entities import DataColumns
from pyspark.sql import SparkSession

class InSparkRepository(SparkRepository):
    def __init__(self,master_name,app_name,data_columns: DataColumns):
        self.spark = SparkSession.builder.master(master_name).appName(app_name).getOrCreate()
        self.data_columns = data_columns

    def read_csv(self, path):
        return self.spark.read.csv(path, header=True, inferSchema=True)

    def write_csv(self, df, path):
        df.toPandas().to_csv(path, index=False, header=True)

    def get_data_columns(self):
        return self.data_columns