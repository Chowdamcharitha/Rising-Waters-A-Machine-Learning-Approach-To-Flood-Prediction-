#loading dataset
import pandas as pd
import numpy as np


# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Loading Data

df=pd.read_excel("dataset/flood dataset.xlsx")

x=df.iloc[:,2:7].values
y=df.iloc[:,10:].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the training data
X_train= scaler.fit_transform(x_train)

# Transform the testing data using the scaler fitted on the training data
X_test = scaler.transform(x_test)

# Check the shapes of the scaled training and testing sets
print("Shape of X_train_scaled:", x_train.shape)
print("Shape of X_test_scaled:", x_test.shape)
from joblib import dump
dump(scaler,"models/transform2.save")
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_accuracy)
from joblib import dump
dump(xgb_model,"models/flood2.save")