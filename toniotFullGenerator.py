import os
import pandas as pd

# Define the features and their data types
string_features = [
    "proto", "service", "conn_state", "dns_query", "ssl_version", "ssl_cipher", 
    "ssl_subject", "ssl_issuer", "http_method", "http_uri", "http_version", 
    "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", 
    "http_user_agent"
]
int_features = [
    "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", 
    "dst_pkts", "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode", 
    "http_trans_depth", "http_request_body_len", "http_response_body_len", 
    "http_status_code"
]
float_features = ["duration"]
boolean_features = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed", 
    "ssl_established", "weird_notice"
]

# Path to the datasets directory
datasets_path = "datasets/toniot_full_network"

# List all CSV files in the datasets directory
csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dataframes = []

# Define columns to drop
columns_to_drop = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'ts']
print(f"Columns to drop: {columns_to_drop}")

# Load and preprocess each CSV file
for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    
    # Read the CSV file
    data = pd.read_csv(os.path.join(datasets_path, csv_file))
    
    # Check and replace non-integer values in 'src_bytes' column
    # non_integer_values = data[~data['src_bytes'].apply(lambda x: str(x).isdigit() or str(x) == '-')]['src_bytes']
    # if not non_integer_values.empty:
    #     print(f"Non-integer values in {csv_file} 'src_bytes' column:")
    #     print(non_integer_values)
    #     data['src_bytes'] = data['src_bytes'].apply(lambda x: 0 if not str(x).isdigit() else int(x))
    #     data.to_csv(os.path.join(datasets_path, csv_file), index=False)
    #     print(f"Updated non-integer values in {csv_file} and saved the file.")
    
    # Drop specified columns
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    print(f"Dropped columns")
    
    # Append the DataFrame to the list
    dataframes.append(data)
    print(f"Appended data from {csv_file} to the list.")

# Concatenate all DataFrames into a single DataFrame
combined_data = pd.concat(dataframes, ignore_index=True)
print("Concatenated all dataframes into a single dataframe.")

# Ensure 'uid' column is not in the combined DataFrame
if 'uid' in combined_data.columns:
    combined_data = combined_data.drop(columns=['uid'])
    print("Dropped 'uid' column from the combined dataframe.")

# Save the combined DataFrame to a new CSV file
print("Saving the combined dataset to a new CSV file.")
combined_data.to_csv(os.path.join('datasets/tonIot', 'combined_dataset.csv'), index=False)
print("Combined dataset created successfully.")
