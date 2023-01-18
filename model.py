import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle

df = pd.read_csv('D:/Data Science Datasets/Rain-Prediction-main/weatherAUS.csv')


numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
discrete_feature = [feature for feature in df.columns if len(df[feature].unique()) < 25]
continous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]

major_null = [feature for feature in df.columns if (df[feature].isnull().sum() * 100 / len(df[feature])) > 30]

def drop_column(feature, df):
    df.drop(feature, axis=1, inplace=True)

for i in major_null:
    drop_column(i, df)

null_all = [feature for feature in df.columns if (df[feature].isnull().sum() * 100 / len(df[feature])) > 0]

for feature in null_all:
    if feature in continous_feature:
        df[feature].fillna(df[feature].median(), inplace=True)
    elif feature in discrete_feature:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

for feature in continous_feature:
    if feature not in df.columns:
        continous_feature.remove(feature)

continous_feature.remove('Sunshine')

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%dT", errors="coerce")
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
df["Date_year"] = df["Date"].dt.year

# To remove outliers, we use 2 methods: Trimming and Capping
# Trimming means delering the whole rows and capping means replacing the value with the upper max value and lower min value
# Here we have used capping
for feature in continous_feature:
    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
    lower_bridge = df[feature].quantile(0.25) - (IQR * 1.5)
    upper_bridge = df[feature].quantile(0.75) + (IQR * 1.5)
    print("Feature name : ", feature, lower_bridge, upper_bridge)
    df.loc[df[feature] >= upper_bridge, feature] = upper_bridge
    df.loc[df[feature] <= lower_bridge, feature] = lower_bridge

df.drop('Date', axis=1, inplace=True)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

for feature in categorical_feature:
    if feature != 'Date':
        df[feature] = label_encoder.fit_transform(df[feature])

df.to_csv("D:/Data Science Datasets/Rain-Prediction-main/PreProcessing_1.csv", index=False)

X = df.drop(["RainTomorrow"], axis=1)
Y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# creating a RF classifier
rf_clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
rf_clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = rf_clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

#Saving the model to file
pickle.dump(rf_clf, open('model.pkl','wb'))

#Loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))