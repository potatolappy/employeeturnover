import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from class_function import *
import pickle

# reading the employee dataset
data = pd.read_csv('D:\DS\Employee Turnover\IBMemployee.csv')

# dropping unnecessary variables to mimic EAZY's database
data = data[["Attrition",
             "Age",
             "YearsWithCurrManager",
             "JobInvolvement",
             "YearsSinceLastPromotion",
             "JobRole",
             "Gender",
             "TotalWorkingYears",
             "NumCompaniesWorked",
             "PerformanceRating",
             "YearsInCurrentRole"]]

# Set train and test set, 80 - 20
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# checking len of train and test dataset
print(train_set)
print(test_set)

# Transform data
prepared_data_train = pipeline_transformer(train_set.drop("Attrition", axis=1))

# Instantiate RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=50, random_state=44)
forest_clf.fit(prepared_data_train, train_set["Attrition"])

# Save the model
with open('D:/forest_model.bin', 'wb') as f_out:
    pickle.dump(forest_clf, f_out)

# import classification report
from sklearn.metrics import classification_report

# Load the test set
X_test = test_set.drop("Attrition", axis=1)
y_test = test_set["Attrition"]

# Load the saved model
with open('D:/forest_model.bin', 'rb') as f_in:
    forest_clf = pickle.load(f_in)

# Transform the test data using the same pipeline
prepared_data_test = pipeline_transformer(X_test)

# Make predictions
predictions = forest_clf.predict(prepared_data_test)

# Generate classification report
report = classification_report(y_test, predictions)

print(report)
