#Data ingetion is reading the dataset
#we use os to create the path


import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components import data_transformations

#data class is use to store class variables, no need for constructer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion process has been started")
        try:
            df = pd.read_csv("C:/Users/raja soni/Desktop/FSDS 2022/Projects/my_regression_gemstone/notebook/data/gemstone.csv")
            logging.info("data set read as pandas data frame")
            
            #Exist_ok means if directory already exist then don't create
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Train test split")
            Train_set,Test_set = train_test_split(df, test_size=0.30, random_state=34)

            logging.info("Ingestion of data is completed")

            Train_set.to_csv(self.ingestion_config.train_data_path, index =False,header = True)
            Test_set.to_csv(self.ingestion_config.test_data_path, index =False,header = True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            


        except Exception as e:
            logging.info("Exception occured at data inegstion part")

            raise CustomException(e, sys)
        
## run the data ingestion 

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_trans = data_transformations.DataTransformation()
    trin_arr, test_arr,_ = data_trans.initiate_data_transformation(train_data_path, test_data_path)