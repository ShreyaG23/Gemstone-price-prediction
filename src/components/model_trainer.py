import os
import sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.utils import save_obj, Evaluate_model
import numpy as np
from dataclasses import dataclass
from src.components import data_transformations
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        model_trainer_config = ModelTrainerConfig()
    
    def Initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting datasets into dependeant and independant data")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            models = {
                "Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()
            }
            Model_report = Evaluate_model(X_train, X_test, y_train, y_test, models)
            print(Model_report)
            logging.info("Model_evaluation completed")
        except Exception as e:
            raise CustomException(e, sys)
