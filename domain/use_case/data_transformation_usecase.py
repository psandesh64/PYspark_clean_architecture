from domain.entity.entities import DataTransformationConfig
from data.spark_repository import InSparkRepository
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,StandardScaler,Imputer

class DataTransformationUseCase:
    def __init__(self, config: DataTransformationConfig, repo: InSparkRepository):
        self.config = config
        self.repo = repo
    
    def _get_missing_value_imputers(self):
        imputers_num = [Imputer(inputCol=column, 
            outputCol=column + "_imputed", 
            strategy="mean") for column in self.repo.get_data_columns().numerical_columns]
        
        return imputers_num
    
    def _get_assembler_impute(self):
        assembler_num = VectorAssembler(
            inputCols=[
                column + "_imputed" for column in self.repo.get_data_columns().numerical_columns
                ],
            outputCol="numerical_features")
        
        return assembler_num
    
    def _get_scaler(self):
        scaler = StandardScaler(
            inputCol="numerical_features", 
            outputCol="scaled_numerical_features", 
            withStd=True, 
            withMean=True)        
        
        return scaler
    
    def _get_string_indexer(self):    
        indexers = [StringIndexer(
            inputCol=column, 
            outputCol=column + "_index", 
            handleInvalid="keep"
            ) for column in self.repo.get_data_columns().categorical_columns]
        
        return indexers

    def _get_oneHot_encode(self):
        encoders = [OneHotEncoder(
            inputCol=column + "_index", 
            outputCol=column + "_encoded"
            ) for column in self.repo.get_data_columns().categorical_columns]
        
        return encoders

    def _get_assembler_categorical(self):
        assembler_cat = VectorAssembler(
            inputCols=[
                column + "_encoded" for column in self.repo.get_data_columns().categorical_columns
                ], 
            outputCol="categorical_features")
        
        return assembler_cat

    def _get_combined_feature_assembler(self):
        combined_assembler = VectorAssembler(
            inputCols=["scaled_numerical_features", "categorical_features"], 
            outputCol="features")

        return combined_assembler

    def initiate_data_transformation(self,train_path,test_path):
        train_df = self.repo.read_csv(train_path)
        test_df = self.repo.read_csv(test_path)
        imputers_num = self._get_missing_value_imputers()
        assembler_num = self._get_assembler_impute()
        scaler = self._get_scaler()
        indexer = self._get_string_indexer()
        encoder = self._get_oneHot_encode()
        assembler_cat = self._get_assembler_categorical()
        assembler_comb = self._get_combined_feature_assembler()

        stages = imputers_num + [assembler_num] + [scaler] + indexer + encoder + [assembler_cat] + [assembler_comb]
        
        # Drop the target column from the train data
        target_column_name=self.repo.get_data_columns().target_column
        input_features_train_df = train_df.select([col for col in train_df.columns if col != target_column_name])
        
        pipeline = Pipeline(stages=stages).fit(input_features_train_df)

        print('Saving model to {}'.format(self.config.preprocessor_obj_file_path))
        pipeline.write().overwrite().save(self.config.preprocessor_obj_file_path)
        print('Model saved...')
        model = PipelineModel.load(self.config.preprocessor_obj_file_path)
        predictions_df = model.transform(test_df)
        predictions_df.show()

    def _get_transformed_df(self,train_path,test_path):
        train_df = self.repo.read_csv(train_path)
        test_df = self.repo.read_csv(test_path)
        model = PipelineModel.load(self.config.preprocessor_obj_file_path)
        train_transformed_df = model.transform(train_df)
        test_transformed_df = model.transform(test_df)

        return train_transformed_df,test_transformed_df