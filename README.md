# Turnover prediction program

this is to backup the ML script
a mix and match from opensource projects, also edited to fit my needs
the file contains:

``` IBMemployee.csv ``` an open source employee dataset that contains attrition status, used for model training, consist of around 20-30 variables

``` IBMemployee - test.csv ``` same source as previous dataset, with attrition column removed

``` class_function.py ``` function to standardize dataset (std scaler and one-hot encoding)

``` model_n_report.py ``` to train the dataset, and generate classification report

``` main.py ``` execute the program

``` forest_model.bin ``` to save the training model, and be used for prediction. It is recommended for each department to have their own model for prediction

# note
1. the model accuracy report indicate better performance on predicting no class than yes class, this might be because of class imbalance, recommended in real life uses to either
   - load more rows, preferrably 10k
   - or make sure the dataset composition include more than 40% of yes class
  need to learn this stuff further

# How to run
1. run the ``` class_function.py ``` to manipulate the dataset
2. run ``` model_n_report.py ``` train and save the model, here you can also do parameter tuning to tweak the accuracy
3. run ``` main.py ``` to execute the program

# Implementation
if you want to deploy this program locally with your own dataset, it is recommended to
1. change the file location of your training data, and create a new model at ``` model_n_report.py ```
2. change which model to use on ``` main.py ```
3. it is recommended to do feature engineering (selecting variables to train the model) on the excel file itself to make the code cleaner and more compact



   
