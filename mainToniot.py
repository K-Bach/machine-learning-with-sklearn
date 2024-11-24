import os
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from libs.utils import print_stats_metrics, getPreprocessor

# List available classifiers
available_classifiers = {
    'randomforest': RandomForestClassifier,
    'svc': svm.SVC,
    'decisiontree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'logisticregression': LogisticRegression,
    'mlp': MLPClassifier
}

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

print("Available classifiers:")
for idx, classifier in enumerate(available_classifiers.keys()):
    print(f"{idx}. {classifier}")

# Let the user choose a classifier
while True:
    classifier_choice = int(input("Select classifier: "))
    if 0 <= classifier_choice < len(available_classifiers):
        break
    else:
        print("Please enter a valid number.")

# Set the classifier
classifier_name = list(available_classifiers.keys())[classifier_choice]
classifier = available_classifiers[classifier_name]()

# List available TonIoT datasets
datasets_path = './datasets/tonIot'
available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
print("Available TonIoT datasets:")
for idx, dataset in enumerate(available_datasets):
    print(f"{idx}. {dataset}")

# Let the user choose a dataset
while True:
    dataset_choice = int(input("Select dataset: "))
    if 0 <= dataset_choice < available_datasets.__len__():
        break
    else:
        print("Please enter a valid number.")
        
chosen_dataset = available_datasets[dataset_choice]

# Create dtype specification for all columns
dtype_spec = {}
for feature in string_features:
        dtype_spec[feature] = 'string'
for feature in int_features:
        dtype_spec[feature] = 'Int64'
for feature in float_features:
        dtype_spec[feature] = 'float64'
for feature in boolean_features:
        dtype_spec[feature] = 'boolean'    

data = pd.read_csv(os.path.join(datasets_path, chosen_dataset), 
                   dtype=dtype_spec,
                   true_values=['T', 't', '1'], false_values=['F', 'f', '0'], 
                   na_values=['-'])
print(f"Loaded {chosen_dataset} dataset.")

# Let the user choose if they want to drop the 'label' or 'type' column
while True:
    drop_column_choice = input("Do you want to drop the 'label' or 'type' column? (Enter 'label' or 'type'): ").strip().lower()
    if drop_column_choice in ['label', 'type']:
        break
    else:
        print("Please enter a valid choice ('label' or 'type').")

# Drop the chosen column
columns_to_drop = [col for col in ['src_ip', 'src_port', 'dst_ip', 'dst_port', drop_column_choice] if col in data.columns]
data = data.drop(columns=columns_to_drop)
print(f"Dropped {drop_column_choice} column.")

# Fill missing values
for col in int_features:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
for col in float_features:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
for col in string_features:
    data[col] = data[col].fillna("")
print("Filled missing values in the train dataset.")

# Separate features and class values
features = data.iloc[:, :-1]
preprocessor = getPreprocessor(features)
x = preprocessor.fit_transform(features)
classes = data.iloc[:, -1]
y = LabelEncoder().fit_transform(classes)
class_names = LabelEncoder().fit(classes).classes_

while True:
    split = input("Split dataset into training and testing sets? (y/n): ")
    if split.lower() == 'y':
        split_dataset = True
        break
    elif split.lower() == 'n':
        split_dataset = False
        break
    else:
        print("Please enter 'y' or 'n'.")

# Split the dataset into training and testing sets
if split_dataset:
    while True:
        try:
            split_percentage = float(input("Enter split percentage for test set (e.g., 0.3 for 30%): "))
            if 0 < split_percentage < 1:
                break
            else:
                print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    print('### Training the model...')
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=split_percentage, 
        random_state=0
        )
    model = classifier.fit(x_train, y_train)
    joblib.dump(model, f"{data.columns[-1]}_{chosen_dataset[:-4]}_{classifier_name}.pkl")
    print('### Testing the model...')
    y_pred = model.predict(x_test)

    print_stats_metrics(y_test, y_pred, class_names)
        
# Train the model on the first dataset and test on a different dataset
else:
    # Train the model on the training dataset
    try:
        model = joblib.load(f"noSplit_{data.columns[-1]}_{chosen_dataset}_{classifier_name}.pkl")
        print('### Model loaded successfully.')
    except FileNotFoundError:
        print('### Training the model...')
        model = classifier.fit(x, y)
        joblib.dump(model, f"noSplit_{data.columns[-1]}_{chosen_dataset}_{classifier_name}.pkl")
    
    print("Available datasets:")
    for idx, dataset in enumerate(available_datasets):
        print(f"{idx}. {dataset}")
    while True:
        test_dataset_choice = int(input("Select testing dataset: "))
        if 0 <= test_dataset_choice < available_datasets.__len__():
                test_data = pd.read_csv(os.path.join(datasets_path, available_datasets[test_dataset_choice]),
                                        dtype=dtype_spec,
                                        true_values=['T', 't', '1'], false_values=['F', 'f', '0'], 
                                        na_values=['-'])
                break
        else:
            print("Please enter a valid number.")
            
    # Drop the chosen column
    columns_to_drop = [col for col in ['src_ip', 'src_port', 'dst_ip', 'dst_port', drop_column_choice] if col in test_data.columns]
    test_data = test_data.drop(columns=columns_to_drop)
    
    # Fill missing values
    for col in int_features:
        test_data[col] = pd.to_numeric(test_data[col], errors="coerce").fillna(0)
    for col in float_features:
        test_data[col] = pd.to_numeric(test_data[col], errors="coerce").fillna(0.0)
    for col in string_features:
        test_data[col] = test_data[col].fillna("")
    print("Filled missing values in the test dataset.")

    features = test_data.iloc[:, :-1]
    preprocessor = getPreprocessor(features)
    x_test = preprocessor.fit_transform(features)
    classes = test_data.iloc[:, -1]
    y_test = LabelEncoder().fit_transform(classes)   
    
    # Test the model on the test dataset
    print('### Testing the model...')
    
    y_pred = model.predict(x_test)
    print_stats_metrics(y_test, y_pred, class_names)

print("### END ###")