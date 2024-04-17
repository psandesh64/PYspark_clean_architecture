from data.spark_repository import InSparkRepository
from domain.entity.entities import DataColumns
from domain.entity.entities import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig
from domain.use_case.data_ingestion_usecase import DataIngestionUseCase
from domain.use_case.data_transformation_usecase import DataTransformationUseCase
from domain.use_case.model_training_usecase import ModelTrainingUseCase

if __name__=="__main__":
    data_columns = DataColumns(
    numerical_columns=["age", "bmi", "children"],
    categorical_columns=["sex", "smoker", "region"],
    target_column="charges"
    )
    repo = InSparkRepository('local','spark_app',data_columns)      # define maste, app name and data columns

    data_ingestion_config = DataIngestionConfig()
    data_ingestion_use_case = DataIngestionUseCase(data_ingestion_config, repo)
    train_data_path, test_data_path = data_ingestion_use_case.execute("artifacts/medical_insurance.csv")

    data_transformation_config = DataTransformationConfig()
    data_transformation_use_case = DataTransformationUseCase(data_transformation_config, repo)
    data_transformation_use_case.initiate_data_transformation(train_data_path, test_data_path)
    train_transformed, test_transformed = data_transformation_use_case._get_transformed_df(train_data_path, test_data_path)

    model_trainer_config = ModelTrainerConfig()
    model_training_use_case = ModelTrainingUseCase(model_trainer_config, repo)
    model_report = model_training_use_case.initiate_model_trainer(train_transformed, test_transformed)