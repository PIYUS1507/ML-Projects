import os
import sys 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logger
from src.utils import save_obj,evaluate_model
from catboost import CatBoostRegressor
from dataclasses import dataclass


@dataclass

class ModelTrainingConfig:
    model_train_file_path=os.path.join("artifact","model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_train_config=ModelTrainingConfig()

    def initiate_model_train(self,train_arr,test_arr):
        try:
            logger.info("the model training has benn started")

            X_train,X_test,Y_Train,Y_Test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            model={
                "Linear Model":LinearRegression(),
                "Random Foresr":RandomForestRegressor(),
                "XGBoost":XGBRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "KNN regressor":KNeighborsRegressor(),
                "Cat boost":CatBoostRegressor(verbose=False),
                "Gradient descent":GradientBoostingRegressor(),
                "Desicion tree":DecisionTreeRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,Y_Train=Y_Train,X_test=X_test,Y_Test=Y_Test,
                                             test_models=model)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=model[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best Model is found")
            
            logger.info("the best model has been found")
            save_obj(
                file_path=self.model_train_config.model_train_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score=r2_score(Y_Test,predicted)

            return score,best_model_name

        except Exception as e:
            raise CustomException(e,sys)

        