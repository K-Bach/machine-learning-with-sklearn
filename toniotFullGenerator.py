import os
import numpy as np
import pandas as pd

# Path to the datasets directory
datasets_path = "datasets/toniot_full_network"

# String features
string_features = [
    "proto", "service", "conn_state",
    "dns_query", "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer",
    "http_method", "http_uri", "http_version", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name", "weird_addl", "http_user_agent"
]
# Number features
int_features = [
    "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode", "http_trans_depth",
    "http_request_body_len", "http_response_body_len", "http_status_code"
]
float_features = ["duration"]
# Boolean features
boolean_features = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed",
    "ssl_established", "weird_notice"
]

# List all CSV files in the datasets directory
csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dataframes = pd.DataFrame()

# Define columns to drop
columns_to_drop = ['http_referrer', 'ts']
print(f"Columns to drop: {columns_to_drop}")

# Load and preprocess each CSV file
for csv_file in csv_files:
    print(f"Processing file: {csv_file}...")
    
    # Read the CSV file
    data = pd.read_csv(os.path.join(datasets_path, csv_file))
    
    # Drop specified columns
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    print(f"Dropped columns")
    
    # Identify rows with infinities in numeric columns
    numeric_cols = int_features + float_features
    inf_rows = data[np.isinf(data[numeric_cols]).any(axis=1)]
    if not inf_rows.empty:
        print("Rows with infinities:")
        print(inf_rows)
    else:
        print("No infinities found in the dataset.")

    # Identify rows with non-numeric values in numeric columns
    for col in numeric_cols:
        if col in data.columns:
            non_numeric_rows = data[~data[col].apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))]
            if not non_numeric_rows.empty:
                print(f"Non-numeric values found in column '{col}':")
                print(non_numeric_rows)
            else:
                print(f"No non-numeric values found in column '{col}'.")
    
    # Append the DataFrame to the list
    combined_data = pd.concat([dataframes, data], ignore_index=True)
    print(f"Appended data from {csv_file} to the list.")

# Ensure 'uid' column is not in the combined DataFrame
if 'uid' in combined_data.columns:
    combined_data = combined_data.drop(columns=['uid'])
    print("Dropped 'uid' column from the combined dataframe.")

# Save the combined DataFrame to a new CSV file
print("Saving the combined dataset to a new CSV file...")
combined_data.to_csv(os.path.join('datasets/tonIot', 'combined_dataset.csv'), index=False)
print("Combined dataset created successfully.")
