import os
import sys
from src.logger import logger
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrain,ModelTrainingConfig


class DataIngestionconfig:
    train_data_path:str=os.path.join('artifact','train.csv')
    test_data_path:str=os.path.join('artifact','test.csv')
    raw_data_path:str=os.path.join('artifact','data.csv')

class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig
    def initiate_data_ingestion(self):
        logger.info("we have enterd in train and split function")

        try:
            df=pd.read_csv(r"notebook\data\stud.csv")
            logger.info("the data has been copied")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logger.info("the directories has been made")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=17)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logger.info("the split has been happened")
            return{
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }


        except Exception as e:
            CustomException(e,sys)


if __name__=="__main__":

    obj=Dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model=ModelTrain()
    print(model.initiate_model_train(train_arr,test_arr))

