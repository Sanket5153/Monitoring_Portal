import mysql.connector
import pandas as pd
import numpy as np

#from predict_pipeline import PredictPipeline, CustomData  # Import PredictPipeline and CustomData

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

pd.set_option('display.max_columns', None)

# Connect to MySQL database
connection = mysql.connector.connect(
    host='localhost',
    user='om',
    password='om',
    database='slurm_acct_db'
)

#if connection.is_connected():
 #   print("Connected to MySQL database")

cursor = connection.cursor()

# Define job ID dynamically
#job_id = 15300
job_id = 26125
#job_id = 15500
# Query with backticks for reserved keywords
query = f"SELECT account, `partition`, priority, timelimit, tres_req, tres_alloc, time_submit, time_start, id_job FROM paramrudra_job_table WHERE id_job={job_id};"
cursor.execute(query)
#print("Job DATA fetched from MySQL database")

# Fetch the selected data
data = cursor.fetchall()

# Store data in a pandas DataFrame
df = pd.DataFrame(data, columns=['account', 'partition', 'priority', 'timelimit', 'tres_req', 'tres_alloc', 'time_submit', 'time_start', 'id_job'])

# Close connection
cursor.close()
connection.close()

# **Preprocessing: Convert categorical values to numerical**
account_mapping = {'nan': 0, 'root': 1, 'cdac': 2, 'aicte': 3, 'nsmhrd': 4, 'tutor': 5, 'mit': 6, 'nsmext': 7}
partition_mapping = {'gpu': 1, 'cpu': 2, 'standard': 3}

# Apply mapping
df['account'] = df['account'].map(account_mapping)
df['partition'] = df['partition'].map(partition_mapping)

# Fill NaN values if any mapping fails
df.fillna(0, inplace=True)

# **Function to process TRES columns (`tres_alloc` & `tres_req`)**
def process_tres_column(column, mapping):
    def process_tres(row):
        extracted_values = {col: np.nan for col in mapping.values()}  # Initialize with NaN
        if pd.notnull(row) and isinstance(row, str):
            for kv in row.split(','):
                if '=' in kv:
                    key, value = kv.split('=')
                    if key in mapping:
                        extracted_values[mapping[key]] = float(value)
        return extracted_values

    return pd.DataFrame(column.apply(process_tres).tolist(), index=column.index)

# Define mappings for `tres_alloc` and `tres_req`
tres_mapping = {
    '1': 'CPU_TRES',
    '2': 'MEMORY_TRES',
    '3': 'ENERGY_TRES',
    '4': 'NODE_TRES',
    '5': 'BILLING_TRES'
}

# Process `tres_alloc`
tres_alloc_df = process_tres_column(df['tres_alloc'], tres_mapping)
tres_alloc_df.columns = [col + "_ALLOC" for col in tres_alloc_df.columns]  # Rename columns

# Process `tres_req`
tres_req_df = process_tres_column(df['tres_req'], tres_mapping)
tres_req_df.columns = [col + "_REQ" for col in tres_req_df.columns]  # Rename columns

# Merge processed columns back into df
df = pd.concat([df, tres_alloc_df, tres_req_df], axis=1)

# Drop original `tres_alloc` and `tres_req` columns
df.drop(columns=['tres_alloc', 'tres_req'], inplace=True)

# Drop the specified columns
df.drop(columns=['MEMORY_TRES_ALLOC', 'ENERGY_TRES_ALLOC', 'ENERGY_TRES_REQ'], inplace=True)

# Convert 'time_start' from Unix timestamp to human-readable format and adjust timezone
df['time_start'] = pd.to_datetime(df['time_start'], unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
df['time_start'] = pd.to_datetime(df['time_start'], errors='coerce')

# Convert 'time_submit' from Unix timestamp to human-readable format and adjust timezone
df['time_submit'] = pd.to_datetime(df['time_submit'], unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
df['time_submit'] = pd.to_datetime(df['time_submit'], errors='coerce')

# Calculate 'Queue_time' in seconds by subtracting 'time_submit' from 'time_start'
df['Queue_time'] = (df['time_start'] - df['time_submit']).dt.total_seconds()

## To print df
#print(df)

# **Prepare the features for prediction**
# The required features should match what the model expects. Using the CustomData class:
custom_data = CustomData(
    account=df['account'].iloc[0],  # Assuming the first row
    partition=df['partition'].iloc[0],
    priority=df['priority'].iloc[0],
    timelimit=df['timelimit'].iloc[0],
    CPU_ALLOC_TRES=df['CPU_TRES_ALLOC'].iloc[0],  # Replace with actual column names after processing
    NODE_ALLOC_TRES=df['NODE_TRES_ALLOC'].iloc[0],
    BILLING_ALLOC_TRES=df['BILLING_TRES_ALLOC'].iloc[0],
    CPU_REQ_TRES=df['CPU_TRES_REQ'].iloc[0],
    MEMORY_REQ_TRES=df['MEMORY_TRES_REQ'].iloc[0],
    NODE_REQ_TRES=df['NODE_TRES_REQ'].iloc[0],
    BILLING_REQ_TRES=df['BILLING_TRES_REQ'].iloc[0],
    Queue_time=df['Queue_time'].iloc[0]
)

# Convert CustomData to DataFrame
df_for_prediction = custom_data.get_data_as_data_frame()

# Use the prediction pipeline to get predictions
predict_pipeline = PredictPipeline()
predictions = predict_pipeline.predict(df_for_prediction)

# Print the prediction result
#print(f"JOB end time Prediction result Log: {predictions}")

predicted_time = np.exp(predictions)
print(f"Predicted JOB END time in seconds: {predicted_time}")
