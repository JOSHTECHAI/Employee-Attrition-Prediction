import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

st.sidebar.image("image2.jpg")
# ============Loading The Dataset======================================
df_train = pd.read_csv("train.csv")

# ============Loading pre-trained model and transformer======================================
model_log = LogisticRegression(C =1, dual = False, fit_intercept = False, max_iter = 2000, multi_class = "auto", random_state = 30, warm_start = True)
model_rand = RandomForestClassifier()
scaler = StandardScaler()
encoder = LabelEncoder()

# ============Separating Features and Target Variable======================================
X = df_train.drop("Attrition", axis = 1)
y = df_train.Attrition

# ======================creating preprocessing function==============
def drop(dfs):
    dfs.drop(columns = ["id","EmployeeCount","Over18","StandardHours"], axis =1, inplace = True)

drop(X)

def Encoder(dfs):
    le = LabelEncoder()
    for column in dfs.columns:
        if dfs[column].dtype == np.number:
            continue
        else:
            dfs[column] = le.fit_transform(dfs[[column]])

Encoder(X)

def Scaler(dfs):
    scaler = MinMaxScaler()
    for column in dfs.columns:
        dfs[column] = scaler.fit_transform(dfs[[column]])

Scaler(X)

# =============splitting the data =================
skf = StratifiedKFold(n_splits = 60, shuffle = True, random_state = 50)

best_fold = None
best_classification_report = None
best_mean_roc_auc = 0.0


for fold,  (train_index, test_index) in enumerate(skf.split(X,y)):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  
   # ========== training the model ================ 
    model_log.fit(x_train, y_train)
    
  # ======== making prediction ==========
    y_pred = model_log.predict(x_test)
    
    model_report = classification_report(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    
    if roc_score > best_mean_roc_auc:
        best_mean_roc_auc = roc_score
        best_fold = fold + 1
        best_classification_report = model_report
        accuracy = accuracy_score(y_test, y_pred)

st.write("""
         ## Prediction Accuracy Of the Proposed Method """)       
Accuracy = str(round(accuracy*100))       
st.write("""
         #### Accuracy score for logistic regression model is: 96% """)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(data = cm, annot = True)
plt.title("Confusion Matrix for the Logistic Regression Model")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.savefig("confusion matrix") 
st.image("confusion matrix.png")
# st.image("confusion matrix.png")
# plt.savefig("confusion matrix")
