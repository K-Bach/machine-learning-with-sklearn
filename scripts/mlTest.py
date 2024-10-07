import pandas as pd
from numpy import genfromtxt
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

##################################################

# Load data using pandas
data = pd.read_csv('../datasets/Train_data.csv')

# Separate features and target
features = data.iloc[:, :-1]
class_value = data.iloc[:, -1]

##################################################

# Encode the target labels
labels = LabelEncoder().fit_transform(class_value)

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
    test_size=0.3, 
    random_state=0
    )

##################################################

# Function to print the metrics
def print_stats_metrics(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred, average='macro'))
    print("Recall: ", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(len(cm))], columns=[f'Predicted {i}' for i in range(len(cm[0]))])
    print("Confusion Matrix: \n", cm_df)

##################################################

# Train the model
pipeline.fit(x_train, y_train)
# Predict the target values
y_pred = pipeline.predict(x_test)

print_stats_metrics(y_test, y_pred)

##################################################
print("###DONE###")