import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logger
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,Y_Train,X_test,Y_Test,test_models):
    try:
            report={}
            logger.info("we entered in Evaluation Model")
            for i in range(len(list(test_models))):
                model=list(test_models.values())[i]

                model.fit(X_train,Y_Train)

                Y_train_pred = model.predict(X_train)
                Y_test_pred=model.predict(X_test)

                train_accuracy=r2_score(Y_Train,Y_train_pred)
                test_accuracy=r2_score(Y_Test,Y_test_pred)

                report[list(test_models.keys())[i]]=test_accuracy
            return report
            
    except Exception as e:
         raise CustomException(e,sys)
        
        
    