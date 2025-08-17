import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def add_time_features(df):
    df = df.copy()
    df['hour_of_day']       = df['purchase_time'].dt.hour
    df['day_of_week']       = df['purchase_time'].dt.day_name()
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds()
    return df

def add_freq_velocity(df):
    df = df.copy().sort_values(['user_id','purchase_time'])
    df['purchase_count']     = df.groupby('user_id')['purchase_time'].transform('count')
    df['time_since_prev']    = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    return df

def get_preprocessor(num_feats, cat_feats):
    # Numeric pipeline: fill NaN → 0, then scale
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: fill NaN → 'Unknown', then one-hot
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])