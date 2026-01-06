import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn. neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" :GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            model_report:dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            # get the best model score
            best_model_score = max(sorted(model_report.values()))
            
            # get best model name
            best_model_name= max(model_report, key=model_report.get)
            best_model=models[best_model_name]
            # best_model=models[best_model_name]
            
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best Found model on both testing and training dataset")
            
            save_object(
                file_path= self.Model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted =  best_model.predict(x_test)
            r2_sc = r2_score(y_test,predicted)
            
            return r2_sc,best_model
            
        except Exception as e:
            raise CustomException(e,sys)


