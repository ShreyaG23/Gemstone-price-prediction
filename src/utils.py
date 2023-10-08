import os
import sys

import numpy as np
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def Evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
    #Predict the target value for train data
            y_pred_train = model.predict(X_train)

    #Predict the target value for train data
            y_pred_test = model.predict(X_test)

            R_score = r2_score(y_test, y_pred_test)
            report[list(models.keys())[i]] = R_score
            print(list(models.keys())[i])
            print("MAE:", mean_absolute_error(y_test, y_pred_test))
            print("MSE:", mean_squared_error(y_test, y_pred_test))
            print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
            print("*"*35)
            print("\n")

            return report
        
    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)



