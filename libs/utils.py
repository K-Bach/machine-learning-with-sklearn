import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Function to print the metrics
def print_stats_metrics(y_test, y_pred, class_names):
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

def getPipeline(features):
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
    return pipeline