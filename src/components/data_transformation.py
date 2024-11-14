import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
#for data transformation ColumnT used for creating pipeline
from sklearn.compose import ColumnTransformer

#This if for handling missing values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler

from src.exception import CustomException
from src.logger import logging
import os 

from src.utils import save_object


#It will take paths
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    #It creates pickle file categorical data to numerical for standard scaler
    def get_data_transformer_object(self):
        '''
            This function is responsible for Data Transformation
        '''
        try:
            numerical_columns=[
                 'id_array_job',
                'id_array_task',
                'id_user',
                'kill_requid',
                'nodes_alloc',
                'cpus_req',
                'derived_ec',
                'exit_code',
                'nodelist',
                'array_max_tasks',
                'array_task_pending',
                'flags',
                'mem_req',
                'priority',
                'state',
                'timelimit',
                'time_submit',
                'time_eligible',
                'time_start',
                'time_end',
                'time_suspended',
                'track_steps',
                'id_job'
            ]

            categorical_columns= [
                
                'gres_req',
                'gres_alloc',
                'constraints',
                'partition',
                'tres_alloc',
                'tres_req',
                'job_type'
            ]

    #---------------------------
        # This is for testing with other dataset 
            # numerical_columns = ["writing_score", "reading_score"]

            # categorical_columns = [
            #     "gender",
            #     "race_ethnicity",
            #     "parental_level_of_education",
            #     "lunch",
            #     "test_preparation_course",
            # ]
    #---------------------------

            ## Its used to handle Missing Values startegy = Median for Outliers
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(missing_values = np.nan,strategy="median")), ## Handling missing values
                    ("scaler",StandardScaler(with_mean=False)) ## Standard scaler
                    # ("scaler",MinMaxScaler())

                ]

            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(missing_values = np.nan,strategy="most_frequent")),
                    #("imputer",SimpleImputer(missing_values = np.nan,strategy='constant', fill_value='unknown')),
                    ("one_hot_encoder",OneHotEncoder(categories='auto', handle_unknown = 'ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                    # ("scaler",MaxAbsScaler())

                ]
            )

            # logging.info("Numerical columns standard scaling completed")

            # logging.info("Categorical columns encoding completed")

            logging.info(f"categorical clumns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")


            ## Combining numerical pipeline with categorical pipeline

            preprocessor=ColumnTransformer(
                [
                    ##num pipline given for numerical_coumns 1.Name 2.Pipeline 3. Cloums 
                ("num_pipeline",num_pipeline,numerical_columns),
                 ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            #this need to be converted to pickle file
            preprocessing_obj=self.get_data_transformer_object()

            #target_column_name="time_end.1"
            target_column_name="time_end"
            numerical_columns=[
                    'id_array_job',
                    'id_array_task',
                    'id_user',
                    'kill_requid',
                    'nodes_alloc',
                    'cpus_req',
                    'derived_ec',
                    'exit_code',
                    'gres_used',
                    'array_max_tasks',
                    'array_task_pending',
                    'flags',
                    'mem_req',
                    'priority',
                    'state',
                    'timelimit',
                    'time_submit',
                    'time_eligible',
                    'time_start',
                    'time_end',
                    'time_suspended',
                    'track_steps',
                    'id_job'
                ]
            
            #-----------------------
            #This is for testing other dataset

            # target_column_name="math_score"
            # numerical_columns = ["writing_score", "reading_score"]


            #------------------------

            #PART REMOVED
            input_feature_train_df=train_df
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df
            # input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
                raise CustomException(e,sys)


