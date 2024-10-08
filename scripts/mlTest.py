import pandas as pd
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import os

# Function to print the metrics
def print_stats_metrics(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred, average='macro'))
    print("Recall: ", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'Actual {class_names[i]}' for i in range(len(cm))], columns=[f'Predicted {class_names[i]}' for i in range(len(cm[0]))])
    cm_df['Total Actual'] = cm_df.sum(axis=1)
    cm_df.loc['Total Predicted'] = cm_df.sum(axis=0)
    print("Confusion Matrix: \n", cm_df)
    # print(cm_df.columns)
    # Save the confusion matrix to a file with a timestamp
    # timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    # cm_filename = f'confusion_matrix_{timestamp}.csv'
    # cm_df.to_csv(cm_filename)
    # print(f"Confusion matrix saved to {cm_filename}")

# List available datasets
datasets_path = '../datasets'
available_datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
print("Available datasets:")
for idx, dataset in enumerate(available_datasets):
    print(f"{idx}. {dataset}")

# Let the user choose a dataset
while True:
    dataset_choice = int(input("Enter the number of the dataset you want to use: "))
    if 0 <= dataset_choice <= available_datasets.__len__():
        break
    else:
        print("Please enter a valid number.")
chosen_dataset = available_datasets[dataset_choice]

# Load the chosen dataset
data = pd.read_csv(os.path.join(datasets_path, chosen_dataset))

# Separate features and target
features = data.iloc[:, :-1]
class_value = data.iloc[:, -1]

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

# Let the user choose the split percentage
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

    # Identify the numerical and categorical features
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    # Create a preprocessor that will scale numerical and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create a pipeline that preprocesses the data and trains the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', svm.SVC())
    ])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, 
        labels, 
        test_size=split_percentage, 
        random_state=0
        )
    # print("Number of 0s in y_test: ", (y_test == 0).sum())

    print('### Training the model...')
    # Train the model
    pipeline.fit(x_train, y_train)
    print('### Testing the model...')
    # Predict the target values
    y_pred = pipeline.predict(x_test)

    print_stats_metrics(y_test, y_pred)
    
else:
    print("Available datasets:")
    for idx, dataset in enumerate(available_datasets):
        print(f"{idx}. {dataset}")
    while True:
        test_dataset_choice = int(input("Enter the number of the dataset you want to use for testing: "))
        if 0 <= dataset_choice <= available_datasets.__len__():
            break
        else:
            print("Please enter a valid number.")
    chosen_test_dataset = available_datasets[dataset_choice]
    test_data = pd.read_csv(os.path.join(datasets_path, chosen_test_dataset))
    test_features = test_data.iloc[:, :-1]

    # Identify the numerical and categorical features
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    # Create a preprocessor that will scale numerical and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', svm.SVC())
    ])

    print('### Training the model...')
    # Train the model
    pipeline.fit(features, class_value)

    print('### Testing the model...')
    # Predict the target values
    y_pred = pipeline.predict(test_features)

    print("Anomalies: ", (y_pred == 'anomaly').sum())
    print("Normal: ", (y_pred == 'normal').sum())
    print("Total: ", y_pred.size)

print("### END ###")