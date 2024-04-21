from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Number pipeline transformer
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object

    '''
    numerics = ['int64']

    num_attrs = data.select_dtypes(include=numerics)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline

# Complete transformation for categorical and numeric data
def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    numerical and categorical data.

    Argument:
        data: original dataframe
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["Gender", "JobRole"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    global full_pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data
