import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")     

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #df=pd.read_csv('notebook\data\scheduler_data.csv')
            df=pd.read_csv('notebook\data\scheduler_data_test.csv')

            categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']

            for feature in categorical_features:
                temp=df.groupby(feature)['time_end'].count()/len(df)
                temp_df=temp[temp>0.01].index
                df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')

            for feature in categorical_features:
                labels_ordered=df.groupby([feature])['time_end'].mean().sort_values().index
                labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
                df[feature]=df[feature].map(labels_ordered)

            #df=pd.read_csv('notebook\data\stud.csv')
            #df=pd.read_csv('notebook\X_train.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    logging.info("Logger has started successfully")
    data_transformation=DataTransformation()
    #data_transformation.initiate_data_transformation(train_data,test_data)
    data_transformation.initiate_data_transformation(train_data,test_data)

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



