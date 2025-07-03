import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.utils import save_obj


@dataclass
class DataTransformConfig:
    preprocessing_obj_file_path = os.path.join('artifact',"preprocessing.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()
    
    def get_data_transform_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("impuetr",SimpleImputer(strategy="most_frequent")),
                    ("encode",OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logger.info(f"categorical data: {categorical_columns}")
            logger.info(f"numerical data: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num pipeline",num_pipeline,numerical_columns),
                    ("categorical pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logger.info("read the cse files from train and test data")

            preprocessor_obj=self.get_data_transform_obj()

            target_column_name="math_score"
            numerical_columns= ['writing_score','reading_score']

            input_feature_train_df=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_data[target_column_name]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logger.info(f"Saved preprocessing object.")
            save_obj(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        