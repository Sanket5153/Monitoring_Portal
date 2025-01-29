import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
#import numpy as np
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


# This part is for Model Trainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

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
            df=pd.read_csv('notebook\data\slurm_job_data.csv')
            #df=pd.read_csv('notebook\data\scheduler_data_test.csv')

            df.drop(columns=['mod_time','job_db_inx','deleted','job_name','id_assoc','id_user','id_group','admin_comment','array_task_str','constraints','container','derived_es','extra','failed_node','id_block','licenses','mcs_label','wckey','system_comment','script_hash_inx','work_dir','submit_line','nodelist','node_inx'],axis=1,inplace=True)

            df.drop(columns=['array_max_tasks','array_task_pending','derived_ec','env_hash_inx','exit_code','flags','id_array_job','id_job','id_array_task','id_qos','id_resv','id_wckey','het_job_id','het_job_offset','kill_requid','state_reason_prev'],axis=1,inplace=True)

            df.drop(columns=['mem_req','state'],axis=1,inplace=True)

            ## Mapping Account column
            df['account']=df['account'].map({'nan':0,'root':1,'cdac':2,'aicte':3,'nsmhrd':4,'tutor':5,'mit':6,'nsmext':7})
            mode_value = df['account'].mode()[0]
            #print(f"Most frequent value (mode): {mode_value}")

            df['account'] = df['account'].fillna(mode_value)

            #----------------------------------------------------------

            ## Mapping partition column

            df['partition']=df['partition'].map({'gpu':1,'cpu':2,'standard':3})
            df.dropna(subset=['tres_alloc'], inplace=True)

            #-------------------------------------------

            ## Mapping TRES Alloc
            key_to_column = {
                '1': 'CPU_ALLOC_TRES',
                '2': 'MEMORY_ALLOC_TRES',
                '3': 'ENERGY_ALLOC_TRES',
                '4': 'NODE_ALLOC_TRES',
                '5': 'BILLING_ALLOC_TRES',
            }

            # Function to process each row and extract values for keys
            def process_tres_column(column, mapping):
                def process_tres(row):
                    # Create a dictionary with NaN values for all keys
                    extracted_values = {col: np.nan for col in mapping.values()}
                    
                    # Only process non-null rows
                    if pd.notnull(row) and isinstance(row, str):
                        key_values = row.split(',')
                        for kv in key_values:
                            if '=' in kv:
                                key, value = kv.split('=')
                                # If the key is in the mapping, add its value
                                if key in mapping:
                                    extracted_values[mapping[key]] = float(value)
                    return extracted_values
                
                # Apply the function to the column
                return column.apply(process_tres)

            # Process the specified column
            processed_df = process_tres_column(df['tres_alloc'], key_to_column)

            # Convert processed results to a DataFrame
            processed_df = pd.DataFrame(list(processed_df.tolist()), index=df.index)

            # Add the processed columns to the original DataFrame
            df = pd.concat([df, processed_df], axis=1)

            df.info()

            #-------------------------------------------------------------

            ## Dropping some columns
            df.drop(columns=['gres_used','MEMORY_ALLOC_TRES'],axis=1,inplace=True)

            #------------------------------------------------------------

            ## TRES REQ conversion
            key_to_column = {
                '1': 'CPU_REQ_TRES',
                '2': 'MEMORY_REQ_TRES',
                '3': 'ENERGY_REQ_TRES',
                '4': 'NODE_REQ_TRES',
                '5': 'BILLING_REQ_TRES',
            }

            # Function to process each row and extract values for keys
            def process_tres_column(column, mapping):
                def process_tres(row):
                    # Create a dictionary with NaN values for all keys
                    extracted_values = {col: np.nan for col in mapping.values()}
                    
                    # Only process non-null rows
                    if pd.notnull(row) and isinstance(row, str):
                        key_values = row.split(',')
                        for kv in key_values:
                            if '=' in kv:
                                key, value = kv.split('=')
                                # If the key is in the mapping, add its value
                                if key in mapping:
                                    extracted_values[mapping[key]] = float(value)
                    return extracted_values
                
                # Apply the function to the column
                return column.apply(process_tres)

            # Process the specified column
            processed_df = process_tres_column(df['tres_req'], key_to_column)

            # Convert processed results to a DataFrame
            processed_df = pd.DataFrame(list(processed_df.tolist()), index=df.index)

            # Add the processed columns to the original DataFrame
            df = pd.concat([df, processed_df], axis=1)


            ## -------------------------

            ## Dropping TRES ALLOC =0 i.e. Failed Jobs

            df.dropna(subset=['CPU_ALLOC_TRES'], inplace=True)

            ## --------------------------------------------

            ## Removing common things

            df.drop(['cpus_req','nodes_alloc','time_suspended','tres_alloc','tres_req','ENERGY_REQ_TRES',],axis=1,inplace=True)
            df.drop(columns=['ENERGY_ALLOC_TRES',],axis=1,inplace=True)

            ##----------------------------------------------------------------

            ## Working on Time Now

            df['time_start'] = pd.to_datetime(df['time_start'], unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['time_start']

            df['time_end'] = pd.to_datetime(df['time_end'], unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['time_end']

            df['time_start'] = pd.to_datetime(df['time_start'], errors='coerce')
            df['time_end'] = pd.to_datetime(df['time_end'], errors='coerce')

            df['total_time'] = (df['time_end'] - df['time_start']).dt.total_seconds()

            df.info() 
            ## Removing Job Total_tIME <= 0

            df.drop(df[df["total_time"] <= 0].index, inplace=True)

            ## -----------------------------------------------------------

            # Queue Time

            df['time_submit'] = pd.to_datetime(df['time_submit'], unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['time_submit']

            df['time_submit'] = pd.to_datetime(df['time_submit'], errors='coerce')
            df['Queue_time'] = (df['time_start'] - df['time_submit']).dt.total_seconds()

            df.drop(columns=['time_start','time_end','time_eligible','time_submit'],axis=1,inplace=True)

            ## ------------------

            ## Log transform for skewed data


            # Log transform highly skewed features
            df['timelimit'] = np.log1p(df['timelimit'])  # log(1 + x) to handle zero values
            df['CPU_ALLOC_TRES'] = np.log1p(df['CPU_ALLOC_TRES'])
            df['NODE_ALLOC_TRES'] = np.log1p(df['NODE_ALLOC_TRES'])
            df['BILLING_ALLOC_TRES'] = np.log1p(df['BILLING_ALLOC_TRES'])
            df['CPU_REQ_TRES'] = np.log1p(df['CPU_REQ_TRES'])
            df['MEMORY_REQ_TRES'] = np.log1p(df['MEMORY_REQ_TRES'])
            df['NODE_REQ_TRES'] = np.log1p(df['NODE_REQ_TRES'])
            df['BILLING_REQ_TRES'] = np.log1p(df['BILLING_REQ_TRES'])
            df['total_time'] = np.log1p(df['total_time'])
            df['Queue_time'] = np.log1p(df['Queue_time'])
            df['priority'] = np.log1p(df['priority'])

            # Check skewness after log transformation
            print(df.skew())

            ## ----------------------------------------------------------

            ## Outliers

            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)

            # Calculate IQR (Interquartile Range)
            IQR = Q3 - Q1

            # Define lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = (df < lower_bound) | (df > upper_bound)

            df = df[~outliers.any(axis=1)]

            # Print the resulting DataFrame after removing outliers
            print("\nDataFrame after removing rows with outliers:")
            print(df)

            # Optionally, print the number of rows removed
            rows_removed = len(outliers) - len(df)
            print(f"\nTotal number of rows removed: {rows_removed}")

            # Print out the rows with outliers (if any)
            print(df[outliers.any(axis=1)])


           ## Removing outliers

            outliers = (df < lower_bound) | (df > upper_bound)
            outlier_counts = outliers.sum()

            # Print out the number of outliers in each column
            #print(outlier_counts)

            ## -------------------------------------------------------------
            ## Checking correleation
            correlation = df.corr()
            print(correlation['total_time'].sort_values(ascending=False))


            

            #-----------END-----------------------------


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

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



