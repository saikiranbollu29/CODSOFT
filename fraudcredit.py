
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import zipfile
import os


with zipfile.ZipFile('fraudTrain.csv.zip') as z:
    with z.open('fraudTrain.csv') as f:
        df_train = pd.read_csv(f)


def preprocess(df):
    df = df.drop(['Unnamed: 0', 'first', 'last', 'street', 'city', 'state',
                  'zip', 'trans_num'], axis=1)


    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df = df.drop('trans_date_trans_time', axis=1)


    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.Timestamp('2020-01-01') - df['dob']).dt.days // 365
    df = df.drop('dob', axis=1)


    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 +
                             (df['long'] - df['merch_long'])**2)
    df = df.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1)


    for col in ['merchant', 'category', 'gender', 'job']:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df

df_train = preprocess(df_train)


X_train_orig = df_train.drop(['is_fraud', 'cc_num'], axis=1)
y_train_orig = df_train['is_fraud']


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_orig, y_train_orig)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train_res)


joblib.dump(clf, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")


