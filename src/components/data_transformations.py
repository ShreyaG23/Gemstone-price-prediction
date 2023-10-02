# Feature engineering
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from src.logger import logging
from sklearn.impute import SimpleImputer # to handle missing values
from sklearn.preprocessing import StandardScaler # for feature scaling (higher values to make it in range)
from sklearn.preprocessing import OrdinalEncoder # For rank type categorical data
from sklearn.pipeline import Pipeline # to connect all the three above mentioned layer(missing value, feature scaling, feature Eng)
from src.utils import save_obj
#to combine all the connect layers we use compose

from sklearn.compose import ColumnTransformer
from src.components import data_ingestion
from src.exception import CustomException

@dataclass
class DataTransfomtionConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransfomtionConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data transformtion initiated")
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cut_categories = ['Fair','Good', 'Very Good', 'Premium', 'Ideal' ]
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info("numerical pipelining initiated")

            numerical_pip = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info("categorical pipelining initiated")
            categorical_pip = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("preprocessor initiated")
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pip', numerical_pip, numerical_cols),
                    ('categorical_pip', categorical_pip, categorical_cols)
                ]
            )
            logging.info("pipelining completed")
            return preprocessor       

        except Exception as e:
            logging.info("Exception occured in data transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # reading the test and train data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"train dataframe head : \n{train_df.head().to_string()}")
            logging.info(f"test dataframe head : \n{test_df.head().to_string()}")

            logging.info("obtaining preprocessor obj")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train_df.drop(labels=drop_columns, axis = 1)
            target_feature_train_df = train_df[[target_column_name]]


            input_feature_test_df = test_df.drop(labels=drop_columns, axis = 1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info("applying preprocessing on trainig and test data")

            #Transformating using preprocessing obj

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessor to train and test datasets.")

            #Entire data covertng into array because we will be able to load array very easy, super fast

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]#C_ - concatanation
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessing_obj 
                )
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )      

                     

        except Exception as e:
            logging.info("Exception occured in the initiate data transformation")
            raise CustomException(e, sys)
        
