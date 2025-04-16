import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ============Loading pre-trained model and transformer======================================
model_log = LogisticRegression()
model_rand = RandomForestClassifier()
scaler = StandardScaler()
encoder = OneHotEncoder()

# ============Loading The Dataset======================================
data = pd.read_csv("train.csv")
shape = data.shape

# ============Dropping id and features with one unique value======================================
data = data.drop(columns = ["id","EmployeeCount","Over18","StandardHours","Education","EnvironmentSatisfaction","WorkLifeBalance","JobLevel","PercentSalaryHike","YearsWithCurrManager","StockOptionLevel","MonthlyRate","RelationshipSatisfaction","DailyRate","HourlyRate","JobSatisfaction"], axis = 1)
data.rename(columns={"DistanceFromHome":"DistanceFromHome (meter)", "MonthlyIncome":"MonthlyIncome ($)"}, inplace=True)
# ============Separating Features and Target Variable======================================
X = data.drop("Attrition", axis = 1)
y = data.Attrition
st.sidebar.image("image1.jpg")

# ============Identifying categorical and numeric value======================================
categorical_features = X.select_dtypes(include = ["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude = ["object"]).columns.tolist()

# ============ Creating transforming for preprocessing======================================
numeric_transfromer = Pipeline(steps=[("scaler", scaler)])
categorical_transfromer = Pipeline(steps=[("encoder", encoder)])

# ============ Using column transformer to apply different transformers to different columns==========
preprocessor = ColumnTransformer(
  transformers= [
    ("num", numeric_transfromer, numeric_features),
    ("cat", categorical_transfromer, categorical_features)
  ]
) 

# ============ Building the full pipeline==================================
pipeline_log = Pipeline(steps=
                    [("preprocessor", preprocessor),
                     ("classifier", model_log)
                     ]
                    )

pipeline_rand = Pipeline(steps=
                    [("preprocessor", preprocessor),
                     ("classifier", model_rand)
                     ]
                    )

# ============ Trainig the Model==================================
pipeline_log.fit(X,y)
pipeline_rand.fit(X,y)

# =============prediction for logistic regression============
def prediction_log():
  st.title("Employee Attrition Prediction With Logistic Regression Model")
  
  # ====== collecting user input=========
  feature_input = {}
  for feature in numeric_features:
    feature_input[feature] = st.number_input(f"Enter {feature}", value = 0)
  for feature in categorical_features:
    unique_values = data[feature].unique() 
    feature_input[feature] = st.selectbox(f"Select {feature}", unique_values)
    
  # =======creating dataframe from user input============
  user_input = pd.DataFrame(data = feature_input, index = [0])
 
#  ========making prediction============
  if st.button("Make Prediction"):
    prediction = pipeline_log.predict(user_input)
    st.subheader("Prediction")
    if prediction[0] == 1:
      st.error("The model predicted that the employee is likely to leave.")
    else:
      st.success("The model predicted that the employee is likely to stayed.")

prediction_log()
    