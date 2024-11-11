import re
import pandas as pd

def extract_slurm_log_data(log_file_path):
    job_id_pattern = re.compile(r'JobId=(\d+)')
    nodelist_pattern = re.compile(r'NodeList=([a-zA-Z0-9\[\],-]+)')
    partition_pattern = re.compile(r'Partition=([a-zA-Z0-9_-]+)')
    user_pattern = re.compile(r'non superuser (\d+) tried to complete batch JobId=|uid (\d+)')  # Combined pattern

    jobs_info = {}
    
    try:
        with open(log_file_path, 'r') as file:
            log_data = file.readlines()

        for line in log_data:
            job_id_match = job_id_pattern.search(line)
            nodelist_match = nodelist_pattern.search(line)
            partition_match = partition_pattern.search(line)
            user_match = user_pattern.search(line)

            # Debug prints
            if user_match:
                user_id = user_match.group(1) if user_match.group(1) else user_match.group(2)
                print(f"Matched UserId: {user_id} in line: {line.strip()}")

            if job_id_match:
                job_id = job_id_match.group(1)
                
                if job_id not in jobs_info:
                    jobs_info[job_id] = {
                        'JobId': job_id,
                        'NodeList': 'N/A',
                        'Partition': 'N/A',
                        'UserId': 'N/A'
                    }

                if nodelist_match:
                    jobs_info[job_id]['NodeList'] = nodelist_match.group(1)
                if partition_match:
                    jobs_info[job_id]['Partition'] = partition_match.group(1)
                if user_match:
                    jobs_info[job_id]['UserId'] = user_id  # Capture user ID

    except FileNotFoundError:
        print(f"Error: The log file '{log_file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return list(jobs_info.values())

# Example usage
log_file_path = './slurmctld.log'
extracted_data = extract_slurm_log_data(log_file_path)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(extracted_data)

# Print or save the extracted data
print(df)
