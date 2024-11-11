import re
import pandas as pd

# Define regular expressions for key parameters
job_regex = r"JobId=(\d+)"
submit_regex = r"SubmitTime=(\d+-\d+-\d+T\d+:\d+:\d+)"
start_regex = r"StartTime=(\d+-\d+-\d+T\d+:\d+:\d+)"
end_regex = r"EndTime=(\d+-\d+-\d+T\d+:\d+:\d+)"
req_mem_regex = r"ReqMem=(\S+)"
max_rss_regex = r"MaxRSS=(\S+)"
num_nodes_regex = r"NumNodes=(\d+)"
num_cpus_regex = r"NumCPUs=(\d+)"
state_regex = r"JobState=(\S+)"
partition_regex = r"Partition=(\S+)"

# Initialize lists to store extracted information
data = []

# Open the log file and extract relevant data
with open('/mnt/data/slurmctld.log', 'r') as file:
    for line in file:
        job_id = re.search(job_regex, line)
        submit_time = re.search(submit_regex, line)
        start_time = re.search(start_regex, line)
        end_time = re.search(end_regex, line)
        req_mem = re.search(req_mem_regex, line)
        max_rss = re.search(max_rss_regex, line)
        num_nodes = re.search(num_nodes_regex, line)
        num_cpus = re.search(num_cpus_regex, line)
        state = re.search(state_regex, line)
        partition = re.search(partition_regex, line)

        if job_id:
            data.append({
                "JobId": job_id.group(1),
                "SubmitTime": submit_time.group(1) if submit_time else None,
                "StartTime": start_time.group(1) if start_time else None,
                "EndTime": end_time.group(1) if end_time else None,
                "ReqMem": req_mem.group(1) if req_mem else None,
                "MaxRSS": max_rss.group(1) if max_rss else None,
                "NumNodes": num_nodes.group(1) if num_nodes else None,
                "NumCPUs": num_cpus.group(1) if num_cpus else None,
                "JobState": state.group(1) if state else None,
                "Partition": partition.group(1) if partition else None
            })

# Convert the data to a DataFrame for easier manipulation
df = pd.DataFrame(data)

# Display the extracted data
print(df.head())
