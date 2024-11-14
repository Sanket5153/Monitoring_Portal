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
        id_array_job: int,
        id_array_task: int,
        id_user: int,
        kill_requid: int,
        nodes_alloc: int,
        nodelist: str,
        cpus_req: int,
        derived_ec: int,
        exit_code: str,
        gres_req: str,
        gres_alloc: str,
        array_max_tasks: int,
        array_task_pending: int,
        constraints: str,
        flags: str,
        mem_req: int,
        partition: str,
        priority: int,
        state: str,
        timelimit: int,
        time_submit: int,
        time_eligible: int,
        time_start: int,
       
        time_suspended: int,
        track_steps: int,
        tres_alloc: str,
        tres_req: str,
        job_type: str,
        id_job: int
        ):
        
    
        self.id_array_job = id_array_job
        self.id_array_task = id_array_task
        self.id_user = id_user
        self.kill_requid = kill_requid
        self.nodes_alloc = nodes_alloc
        self.nodelist = nodelist
        self.cpus_req = cpus_req
        self.derived_ec = derived_ec
        self.exit_code = exit_code
        self.gres_req = gres_req
        self.gres_alloc = gres_alloc
        self.array_max_tasks = array_max_tasks
        self.array_task_pending = array_task_pending
        self.constraints = constraints
        self.flags = flags
        self.mem_req = mem_req
        self.partition = partition
        self.priority = priority
        self.state = state
        self.timelimit = timelimit
        self.time_submit = time_submit
        self.time_eligible = time_eligible
        self.time_start = time_start
        
        self.time_suspended = time_suspended
        self.track_steps = track_steps
        self.tres_alloc = tres_alloc
        self.tres_req = tres_req
        self.job_type = job_type
        self.id_job = id_job

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "id_array_job": [self.id_array_job],
                "id_array_task": [self.id_array_task],
                "id_user": [self.id_user],
                "kill_requid": [self.kill_requid],
                "nodes_alloc": [self.nodes_alloc],
                "nodelist": [self.nodelist],
                "cpus_req": [self.cpus_req],
                "derived_ec": [self.derived_ec],
                "exit_code": [self.exit_code],
                "gres_req": [self.gres_req],
                "gres_alloc": [self.gres_alloc],
                "array_max_tasks": [self.array_max_tasks],
                "array_task_pending": [self.array_task_pending],
                "constraints": [self.constraints],
                "flags": [self.flags],
                "mem_req": [self.mem_req],
                "partition": [self.partition],
                "priority": [self.priority],
                "state": [self.state],
                "timelimit": [self.timelimit],
                "time_submit": [self.time_submit],
                "time_eligible": [self.time_eligible],
                "time_start": [self.time_start],
                
                "time_suspended": [self.time_suspended],
                "track_steps": [self.track_steps],
                "tres_alloc": [self.tres_alloc],
                "tres_req": [self.tres_req],
                "job_type": [self.job_type],
                "id_job": [self.id_job]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
