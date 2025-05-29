import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# This part is for Model Trainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI if needed
            db = client["slurm"]  # Replace with your database name
            collection = db["job_data"]  # Replace with your collection name

            # Fetch all data from the MongoDB collection
            cursor = collection.find()  # You can add filters here if needed
            df = pd.DataFrame(list(cursor))  # Convert MongoDB data to DataFrame

            # If MongoDB returns nested dictionaries, you might need to normalize them
            # Example:
            df = pd.json_normalize(list(cursor))

            # Preprocess categorical features
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

            for feature in categorical_features:
                temp = df.groupby(feature)['time_end'].count() / len(df)
                temp_df = temp[temp > 0.01].index
                df[feature] = np.where(df[feature].isin(temp_df), df[feature], 'Rare_var')

            for feature in categorical_features:
                labels_ordered = df.groupby([feature])['time_end'].mean().sort_values().index
                labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
                df[feature] = df[feature].map(labels_ordered)

            # Log the data reading
            logging.info('Read the dataset from MongoDB as dataframe')

            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split data to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    logging.info("Logger has started successfully")
    data_transformation = DataTransformation()
    #data_transformation.initiate_data_transformation(train_data, test_data)
    data_transformation.initiate_data_transformation(train_data, test_data)

    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
