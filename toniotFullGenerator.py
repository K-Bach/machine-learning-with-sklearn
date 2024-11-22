import os
import pandas as pd

# Path to the datasets directory
datasets_path = "datasets/toniot_full_network"

# List all CSV files in the datasets directory
csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dataframes = pd.DataFrame()

# Define columns to drop
columns_to_drop = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'ts']
print(f"Columns to drop: {columns_to_drop}")

# Load and preprocess each CSV file
for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    
    # Read the CSV file
    data = pd.read_csv(os.path.join(datasets_path, csv_file))
    
    # Drop specified columns
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    print(f"Dropped columns")
    
    # Append the DataFrame to the list
    combined_data = pd.concat([dataframes, data], ignore_index=True)
    print(f"Appended data from {csv_file} to the list.")

# Ensure 'uid' column is not in the combined DataFrame
if 'uid' in combined_data.columns:
    combined_data = combined_data.drop(columns=['uid'])
    print("Dropped 'uid' column from the combined dataframe.")

# Save the combined DataFrame to a new CSV file
print("Saving the combined dataset to a new CSV file.")
combined_data.to_csv(os.path.join('datasets/tonIot', 'combined_dataset.csv'), index=False)
print("Combined dataset created successfully.")
