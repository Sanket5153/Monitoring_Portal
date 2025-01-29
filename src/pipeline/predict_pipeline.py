import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            #model_path=os.path.join("artifacts","model.pkl") ## Model path
            #preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self,
        account: float,
        partition: int,
        priority: float,
        timelimit: float,
        CPU_ALLOC_TRES: float,
        NODE_ALLOC_TRES: float,
        BILLING_ALLOC_TRES: float,
        CPU_REQ_TRES: float,
        MEMORY_REQ_TRES: float,
        NODE_REQ_TRES: float,
        BILLING_REQ_TRES: float,
       # total_time: float,
        Queue_time: float,
        ):
        
        self.account = account
        self.partition = partition
        self.priority = priority
        self.timelimit = timelimit
        self.CPU_ALLOC_TRES = CPU_ALLOC_TRES
        self.NODE_ALLOC_TRES = NODE_ALLOC_TRES
        self.BILLING_ALLOC_TRES = BILLING_ALLOC_TRES
        self.CPU_REQ_TRES = CPU_REQ_TRES
        self.MEMORY_REQ_TRES = MEMORY_REQ_TRES
        self.NODE_REQ_TRES = NODE_REQ_TRES
        self.BILLING_REQ_TRES = BILLING_REQ_TRES
        #self.total_time = total_time
        self.Queue_time = Queue_time

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "account": [self.account],
                "partition": [self.partition],
                "priority": [self.priority],
                "timelimit": [self.timelimit],
                "CPU_ALLOC_TRES": [self.CPU_ALLOC_TRES],
                "NODE_ALLOC_TRES": [self.NODE_ALLOC_TRES],
                "BILLING_ALLOC_TRES": [self.BILLING_ALLOC_TRES],
                "CPU_REQ_TRES": [self.CPU_REQ_TRES],
                "MEMORY_REQ_TRES": [self.MEMORY_REQ_TRES],
                "NODE_REQ_TRES": [self.NODE_REQ_TRES],
                "BILLING_REQ_TRES": [self.BILLING_REQ_TRES],
                #"total_time": [self.total_time],
                "Queue_time": [self.Queue_time],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
