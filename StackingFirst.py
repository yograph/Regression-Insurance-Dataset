import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import warnings
import os
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.compose import ColumnTransformer
from fancyimpute import IterativeSVD

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import warnings
import os

# Models
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

# Neural Network
from sklearn.neural_network import MLPRegressor  # or you can use Keras/PyTorch

# Utilities
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import KNNImputer
from fancyimpute import IterativeSVD

# For hyperparameter tuning
import optuna

# If you need to create a submission file:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

input_files = {}
for dirname, _, filenames in os.walk(r'C:\Users\youse\Desktop\Regression_insurance'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if file_path.endswith('.csv'):
            print( f'Importing {file_path}')
            input_files[filename] = pd.read_csv( file_path)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = input_files['train.csv'].drop(columns='id')
df_test = input_files['test.csv']
df_submission = input_files['sample_submission.csv']
df_train.info()
# You can do the train set that can show all the NaN values in the dataset
pd.concat((df_train.isna().sum().sort_values(ascending=False).rename('#'), df_train.isna().mean().mul(100).round(2).rename('%')), axis=1) 

# Split the data to Categorical and Numerical 
ordinal_variables = {
    'Customer Feedback' : ['Poor', 'Average', 'Good'],
    'Marital Status' : ['Married', 'Divorced', 'Single'],
    'Location': ['Urban', 'Suburban', 'Rural'],
    'Policy Type' : ['Premium', 'Comprehensive', 'Basic'],
    'Gender' : ['Male', 'Female'],
    'Education Level': ['High School', "Bachelor's", "Master's", 'PhD'],
    'Smoking Status' : ['No', 'Yes'],
    'Exercise Frequency' : ['Daily', 'Weekly', 'Monthly', 'Rarely'],
    'Property Type' : ['House', 'Condo', 'Apartment'],
    'Occupation': ['Unemployed','Self-Employed','Employed'],
}

def transform_policy_date(df):
    dt = pd.to_datetime( df['Policy Start Date'])
    df['start_date_month'] = dt.dt.month
    df['start_date_dow'] = dt.dt.dayofweek # let's see what this thingy is
    df['start_date_year'] = dt.dt.year
    return df#.drop(columns=['Policy Start Date'])


df_train = transform_policy_date(df_train)
df_test = transform_policy_date(df_test)
df_train.head()

numerical_variables = df_train.drop(columns='Premium Amount').select_dtypes(exclude=('category', object)).columns.tolist()
print('Numerical variables:', numerical_variables)
print('Ordinal variables:', list(ordinal_variables.keys()))

#



# Drop 'Policy Start Date' column
if 'Policy Start Date' in df_train.columns or 'Policy Start Data' in df_test.columns:
    df_train.drop('Policy Start Date', axis=1, inplace=True)
    df_test.drop('Policy Start Date', axis=1, inplace=True)
else:
    print("Not there")

# 1) Create dummy variables
df_train_start = pd.get_dummies(
    df_train, 
    columns=[column for column in list(ordinal_variables.keys())], 
    prefix=[column for column in list(ordinal_variables.keys())]
)
df_test_start = pd.get_dummies(
    df_test,
    columns=[column for column in list(ordinal_variables.keys())],
    prefix=[column for column in list(ordinal_variables.keys())]
)
print("This is before the alignment")
print(df_train_start.columns)
print(df_test_start.columns)


# 2) Align the test columns with the train columns
df_test_start = df_test_start.reindex(columns=df_train_start.columns, fill_value=0)
print("This is after the alignment")
print(df_train_start.columns)
print(df_test_start.columns)



for variable in numerical_variables:
    print(variable)
df_train_start.head()



print("I wanna die")

# Initialize the KNN imputer
#imputer = KNNImputer(n_neighbors=2)
svd_imputer = IterativeSVD(max_iters=10)


# Apply the imputer to the entire training dataset
# Impute training dataset with IterativeSVD
df_train_imputed = pd.DataFrame(svd_imputer.fit_transform(df_train_start), columns=df_train_start.columns)
print("Training dataset imputed with IterativeSVD.")

# Use KNNImputer for the test dataset
print("Imputing test dataset with IterativeSVD...")
df_test_imputed = pd.DataFrame(svd_imputer.fit_transform(df_test_start), columns=df_test_start.columns)
print("Test dataset imputed with IterativeSVD.")

# The test set (with the same transformations) is:
X_test = df_test_imputed


# Assuming that df_train_imputed includes "Premium Amount" as the target and df_test_imputed is the test set without Premium Amount.

# Separate features (X) and target (y) in the training data
X = df_train_imputed.drop('Premium Amount', axis=1)
y = df_train_imputed['Premium Amount']

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

# 1. Base Models
catboost_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, verbose=0)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)

# 2. Meta-Model
meta_model = LinearRegression()  

# 3. Create Stacking Model
stacking_model = StackingRegressor(
    estimators=[
        ('catboost', catboost_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=meta_model,  # Combines predictions
    passthrough=False  # Set True to pass original features as well
)

# 4. Train the Stacking Model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
stacking_model.fit(X_train, y_train)

# 5. Evaluate the Model
y_val_pred = stacking_model.predict(X_val)
y_val_pred[y_val_pred < 0] = 0  # Clip negative predictions if necessary
msle = mean_squared_log_error(y_val, y_val_pred)
print("Stacking Model MSLE:", msle)

# 6. Final Test Predictions
test_preds = stacking_model.predict(X_test)
test_preds[test_preds < 0] = 0

print(test_preds)
df_submission['Premium Amount'] = test_preds
df_submission.to_csv('submission.csv', index=False)
print("Ensemble (Neural Net) submission file 'submission.csv' created.")


