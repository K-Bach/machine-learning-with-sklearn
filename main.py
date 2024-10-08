import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from libs.utils import print_stats_metrics, getPipeline
import os

# List available datasets
datasets_path = './datasets'
available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
print("Available datasets:")
for idx, dataset in enumerate(available_datasets):
    print(f"{idx}. {dataset}")

# Let the user choose a dataset
while True:
    dataset_choice = int(input("Enter the dataset you want to use: "))
    if 0 <= dataset_choice < available_datasets.__len__():
        break
    else:
        print("Please enter a valid number.")
chosen_dataset = available_datasets[dataset_choice]

# Load the chosen dataset
data = pd.read_csv(os.path.join(datasets_path, chosen_dataset))

# Separate features and class values
features = data.iloc[:, :-1]
class_value = data.iloc[:, -1]

# Create a pipeline
pipeline = getPipeline(features)

while True:
    split = input("Do you want to split the dataset into training and testing sets? (y/n): ")
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
            split_percentage = float(input("Enter the split percentage for the test set (e.g., 0.3 for 30%): "))
            if 0 < split_percentage < 1:
                break
            else:
                print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Encode the target labels
    labels = LabelEncoder().fit_transform(class_value)
    class_names = LabelEncoder().fit(class_value).classes_

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, 
        labels, 
        test_size=split_percentage, 
        random_state=0
        )
    # print("Number of 0s in y_test: ", (y_test == 0).sum())

    print('### Training the model...')
    pipeline.fit(x_train, y_train)
    print('### Testing the model...')
    y_pred = pipeline.predict(x_test)

    print_stats_metrics(y_test, y_pred, class_names)
    
# Train the model on the first dataset and test on a different dataset
else:
    print("Available datasets:")
    for idx, dataset in enumerate(available_datasets):
        print(f"{idx}. {dataset}")
    while True:
        test_dataset_choice = int(input("Enter the dataset you want to use for testing: "))
        if 0 <= test_dataset_choice < available_datasets.__len__():
            break
        else:
            print("Please enter a valid number.")

    chosen_test_dataset = available_datasets[test_dataset_choice]
    test_data = pd.read_csv(os.path.join(datasets_path, chosen_test_dataset))

    print('### Training the model...')
    pipeline.fit(features, class_value)
    print('### Testing the model...')
    y_pred = pipeline.predict(test_data)

    print("Anomalies: ", (y_pred == 'anomaly').sum())
    print("Normal: ", (y_pred == 'normal').sum())
    print("Total: ", y_pred.size)

print("### END ###")