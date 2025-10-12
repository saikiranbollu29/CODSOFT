import pandas as pd
import numpy as np
import joblib
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

clf = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

print(" Model and scaler loaded successfully!")

with zipfile.ZipFile('fraudTest.csv.zip') as z:
    with z.open('fraudTest.csv') as f:
        df_test = pd.read_csv(f)
print(" Test dataset loaded:", df_test.shape)
df_test.drop(['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip',
              'trans_num'], axis=1, inplace=True)

df_test['trans_date_trans_time'] = pd.to_datetime(df_test['trans_date_trans_time'])
df_test['hour'] = df_test['trans_date_trans_time'].dt.hour
df_test['day_of_week'] = df_test['trans_date_trans_time'].dt.dayofweek
df_test.drop('trans_date_trans_time', axis=1, inplace=True)

df_test['dob'] = pd.to_datetime(df_test['dob'])
df_test['age'] = (pd.Timestamp('2020-01-01') - df_test['dob']).dt.days // 365
df_test.drop('dob', axis=1, inplace=True)

df_test['distance'] = np.sqrt((df_test['lat'] - df_test['merch_lat'])**2 +
                              (df_test['long'] - df_test['merch_long'])**2)

df_test.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1, inplace=True)

for col in ['merchant', 'category', 'gender', 'job']:
    df_test[col] = LabelEncoder().fit_transform(df_test[col])

X_test = df_test.drop(['is_fraud', 'cc_num'], axis=1)
y_test = df_test['is_fraud']

X_test_scaled = scaler.transform(X_test)

y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]
print("\n Test Data Evaluation:")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
