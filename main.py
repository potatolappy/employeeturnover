import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from class_function import *
import pandas as pd

# landing page
print("=" * 40)
print(" Employee Turnover Prediction - Version April 2024")
print(" Please wait before selecting your file")
print("=" * 40)

def predict_attrition(config=None, model=None):
    '''
    Function to load data, transform and predict
    argument:
        config: tkinter will prompt file selection, employee csv, store in separate variable
        model: a trained Random Forest Classifier bin
        both will be automatically selected
    return:
        y_pred: a prediction that has a feature importance & summary of prediction
    '''
    if config is None:
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        print("Loading file: " + filename)
        attrition_df_temp = pd.read_csv(filename)

    if model is None:
        with open('D:/forest_model.bin', 'rb') as file:
            model = pickle.load(file)
            file.close()

    # checking loaded file
    print(attrition_df_temp)

    # selecting variables inside the dataset
    data_select = attrition_df_temp[
        ["Age",
         "YearsWithCurrManager",
         "JobInvolvement",
         "YearsSinceLastPromotion",
         "JobRole",
         "Gender",
         "TotalWorkingYears",
         "NumCompaniesWorked",
         "PerformanceRating",
         "YearsInCurrentRole"]].copy()
    data_select = pipeline_transformer(data_select)
    y_pred = model.predict(data_select)

    # Calculate percentage of 'Yes' predictions
    yes_count = sum(1 for prediction in y_pred if prediction == 'Yes')
    total_count = len(y_pred)
    percentage_yes = (yes_count / total_count) * 100

    # print prediction summary
    print("-" * 40)
    print("Attrition Prediction:")
    print(f"Prediction - Employee at risk of resignations: {percentage_yes:.2f}%")
    print(f"Total predicted at risk: {yes_count} people")
    print("-" * 40)

    # Feature Importance Analysis
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_

        # features list
        feature_names = ["Age", "YearsWithCurrManager", "JobInvolvement", "YearsSinceLastPromotion",
                         "JobRole", "Gender", "TotalWorkingYears", "NumCompaniesWorked",
                         "PerformanceRating", "YearsInCurrentRole"]

        # dict and zip the features with values
        importance_dict = dict(zip(feature_names, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        # print Features Importance ranking
        print("\nRanking - Top Driving Reasons for Resignations:")
        print("-" * 40)
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")

    return y_pred

# Run the program
predict_attrition()
