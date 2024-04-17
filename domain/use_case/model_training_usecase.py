from domain.entity.entities import ModelTrainerConfig
from data.spark_repository import InSparkRepository
from pyspark.ml.regression import (
    RandomForestRegressor,
    DecisionTreeRegressor,
    GBTRegressor,
    LinearRegression
)
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

class ModelTrainingUseCase:
    def __init__(self, config: ModelTrainerConfig, repo: InSparkRepository):
        self.config = config
        self.repo = repo

    def _get_models(self):
        label_col = self.repo.get_data_columns().target_column
        models = {
                "Random Forest": RandomForestRegressor(featuresCol="features",labelCol=label_col),
                "Decision Tree": DecisionTreeRegressor(featuresCol="features",labelCol=label_col),
                "Gradient Boosting": GBTRegressor(featuresCol="features",labelCol=label_col),
                "Linear Regression": LinearRegression(featuresCol="features",labelCol=label_col,regParam=0.01, maxIter=100)
            }
        
        return models
    
    def _get_paramGrid(self):
        paramGrid = {
                "Random Forest": ParamGridBuilder().addGrid(RandomForestRegressor.numTrees, [10, 20, 30]).build(),
                "Decision Tree": ParamGridBuilder().addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15]).build(),
                "Gradient Boosting": ParamGridBuilder().addGrid(GBTRegressor.maxDepth, [2, 4, 6]).build(),
                "Linear Regression": ParamGridBuilder().build()
            }
        return paramGrid

    def initiate_model_trainer(self, train_transformed_df, test_transformed_df):
        # Train models
        evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol=self.repo.get_data_columns().target_column,
            metricName="r2")
        print("initiating model training")
         ## Train and evaluate using cross-validation
        model_report = {}
        for model_name, model in self._get_models().items():
            param_grid = self._get_paramGrid()[model_name]
            crossval = CrossValidator(estimator=model,
                                    estimatorParamMaps=param_grid,
                                    evaluator=evaluator,
                                    numFolds=5)
            
            cv_model = crossval.fit(train_transformed_df)
            predictions = cv_model.transform(test_transformed_df)
            model_report[model_name] = evaluator.evaluate(predictions)  # r2_score

        print("Model Evaluation report(r2 score):",model_report)

        # Retrieve the best model
        best_model_name = max(model_report, key=model_report.get)
        best_model = self._get_models()[best_model_name]

        print(model_report[best_model_name],best_model_name)  

        regressor = best_model.fit(train_transformed_df)

        file_path = self.config.train_model_file_path
        regressor.write().overwrite().save(file_path)

        return model_report[best_model_name]
