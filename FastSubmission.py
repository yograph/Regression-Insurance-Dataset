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

path = r'C:\Users\youse\Desktop\Regression_insurance'

import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(30)

kaggle_path = '/kaggle/input/playground-series-s4e12'
import os
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print (f'content kaggle_path : {kaggle_path}')
train_df = pl.scan_csv(f'{kaggle_path}/train.csv').collect()

test_df = pl.scan_csv(f'{kaggle_path}/test.csv').collect()
sample_df = pl.scan_csv(f'{kaggle_path}/sample_submission.csv').collect()

display (train_df.collect_schema())

display (train_df.head(5))

def data_clean (raw : pl.DataFrame) -> pl.DataFrame :
    drop_columns = ['Policy Start Date']
    result = raw.with_columns(pl.col('Policy Start Date').str.head (10).alias('Policy Start Day'))
    if ('Premium Amount' in raw.columns) :
        result = result.filter (~pl.col('Premium Amount').is_nan())
        result = result.filter (pl.col('Premium Amount') > 0)
        result = result.with_columns(pl.col('Premium Amount').log().alias('Premium log'))
        drop_columns.append ('Premium Amount')
    if ('id' in raw.columns) :    
        drop_columns.append ('id')
    result = result.drop(drop_columns)
    return result

train_clean_df = data_clean (train_df)
test_clean_df = data_clean (test_df)

from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(path = '/kaggle/working/Autogluon/',
                                      label='Premium log', 
                              problem_type = 'regression', 
                              eval_metric = 'root_mean_squared_error',  
#                              sample_weight = 'my_weight',
                              )

    
predictor.fit(train_data= train_clean_df.to_pandas(), 
                       # dynamic_stacking=False, num_stack_levels=1,
                       presets='best_quality',
# best_quality,  medium_quality                         
                       time_limit = 33000, 
                       num_gpus=1, 
                       num_bag_folds = 7, 
                       num_stack_levels = 4, 
                       auto_stack = True, 
                       dynamic_stacking=True   
                       )


predictor.leaderboard()
print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
predictor.fit_summary ()

# this cell allows to reload the trained model from disk
predictor = TabularPredictor.load("/kaggle/working/Autogluon")

predictions_log = pl.Series (predictor.predict(test_clean_df.to_pandas () ))

# trainsforming the results back from the log scake 
predictions = predictions_log.exp()
predictions

submission = sample_df.with_columns (predictions.alias('Premium Amount'))
submission

submission.write_csv('submission.csv')

import zipfile

def zip_files_in_directory(directory, zip_name):
    num_files_deleted = 0 
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))
                os.remove(file_path)
                num_files_deleted += 1
    return num_files_deleted  
# Example usage
directory = '/kaggle/working/Autogluon'
zip_name = 'Autogluon.zip'
n = zip_files_in_directory(directory, zip_name)

print(f"deleted {n} files in {directory}")

