import os
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from libs.utils import print_stats_metrics, getPipeline

# List available classifiers
available_classifiers = {
    'randomforest': RandomForestClassifier,
    'svc': svm.SVC,
    'decisiontree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'logisticregression': LogisticRegression,
    'mlp': MLPClassifier
}

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

# Create dtype specification for all columns
dtype_spec = {}
for feature in string_features:
        dtype_spec[feature] = 'str'
for feature in int_features:
        dtype_spec[feature] = 'int'
for feature in float_features:
        dtype_spec[feature] = 'float'
for feature in boolean_features:
        dtype_spec[feature] = 'bool'

data = pd.read_csv(os.path.join(datasets_path, chosen_dataset))
print(f"Loaded {chosen_dataset} dataset.")

# Let the user choose if they want to drop the 'label' or 'type' column
while True:
    drop_column_choice = input("Do you want to drop the 'label' or 'type' column? (Enter 'label' or 'type'): ").strip().lower()
    if drop_column_choice in ['label', 'type']:
        break
    else:
        print("Please enter a valid choice ('label' or 'type').")

# Drop the chosen column
columns_to_drop = ['src_ip', 'src_port', 'dst_ip', 'dst_port', drop_column_choice]
data = data.drop(columns=columns_to_drop)

# Fill missing values
for feature in string_features:
    if feature in data.columns:
        data[feature] = data[feature].replace("-", "")
for feature in int_features:
    if feature in data.columns:
        data[feature] = data[feature].replace("-", 0)
for feature in boolean_features:
    if feature in data.columns:
        data[feature] = data[feature].replace("-", False)
for feature in float_features:
    if feature in data.columns:
        data[feature] = data[feature].replace("-", 0.0)
        
# Assign correct dtypes
for feature in string_features:
    if feature in data.columns:
        data[feature] = data[feature].astype(str)
for feature in int_features:
    if feature in data.columns:
        data[feature] = data[feature].astype(int)
for feature in float_features:
    if feature in data.columns:
        data[feature] = data[feature].astype(float)
for feature in boolean_features:
    if feature in data.columns:
        data[feature] = data[feature].astype(bool)
print("Data types assigned.")

# Separate features and class values
features = data.iloc[:, :-1]
class_value = data.iloc[:, -1]

# Create a pipeline
pipeline = getPipeline(features, classifier)
print("Pipeline created.")
split_dataset = True

# Encode the target labels
labels = LabelEncoder().fit_transform(class_value)
class_names = LabelEncoder().fit(class_value).classes_

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.3, 
    random_state=0
    )

print('### Training the model...')
pipeline.fit(x_train, y_train)
print('### Testing the model...')
y_pred = pipeline.predict(x_test)

print_stats_metrics(y_test, y_pred, class_names)
  
print("### END ###")