import sys
from src.exception import CustomException
from src.logger import logger
import pandas as pd
from src.utils import load_obj
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path=os.path.join('artifact','model.pkl')
        preprocessor_path=os.path.join('artifact','preprocessing.pkl')

        print('before loading')

        model = load_obj(model_path)
        preprocessor=load_obj(preprocessor_path)

        print('after loading')

        data_scaled=preprocessor.transform(features)
        result=model.predict(data_scaled)

        return result




class CustomData:
    def __init__(self,
                 gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def get_data_data_frame(self):
        try:
            custome_data_struct={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            Df=pd.DataFrame(custome_data_struct)
            logger.info("the data frame has been found")
            return Df

        except Exception as e:
            raise CustomException(e,sys)
            

        