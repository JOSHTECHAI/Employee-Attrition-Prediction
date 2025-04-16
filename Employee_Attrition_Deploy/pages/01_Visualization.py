import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()
encoder = LabelEncoder()

# ============Loading The Dataset======================================
df_train = pd.read_csv("train.csv")

# ============Separating Features and Target Variable======================================
X = df_train.drop("Attrition", axis = 1)
y = ["leave" if i == 1 else "stay" for i in df_train.Attrition]

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
    scaler = StandardScaler()
    for column in dfs.columns:
        dfs[column] = scaler.fit_transform(dfs[[column]])

Scaler(X)

st.sidebar.image("image3.jpg")


# ============Creating Visualizing App==================================
def visualize_title():
  st.title("Exploratory Data Analysis")
  st.write("""
           ##### Visualizing and Analysing features which incldes; Age, Department, Gender,Marital Status, Years at Company, and Explanation of their effect on the Attrition of Employee.
           """)

visualize_title()

def age_visualize():
  st.write("**Age Distribution**")
  plt.figure(figsize = (11,5))
  sns.countplot(data = df_train, x = "Age", hue = y, palette = "colorblind")
  plt.title("Age of Employee against Attrition in the Company")
  plt.xticks(rotation = 70)
  plt.tight_layout()
  plt.show()
  plt.savefig("Age Distribution")
  st.image("Age Distribution.png")
  
age_visualize()

def dept_visualize():
  st.write("**Department Distribution**")
  label = [1,2,0]
  x = df_train["Department"].value_counts()
  labels =[]
  for i in label:
    if i==0:
        labels.append("Human Resources")
    elif i == 1:
        labels.append("Research & Development")
    elif i == 2:
        labels.append("Sales")
        
  plt.figure(figsize = (6,4))
  plt.pie(x= x, labels = labels, autopct = '%1.2f%%', explode = [0,0,0])
  plt.title("Department with the Higest Percentage of Employees")
  plt.tight_layout()
  plt.show()
  plt.savefig("dept distribution")
  st.image("dept distribution.png")

  st.write("**How Department Distribution Affect the Level of Attrition**")
  plt.figure(figsize = (12,5))
  sns.countplot(data = df_train, x = "Department", hue = y, palette = "colorblind")
  plt.title("Department of Employee in the Company")
  plt.xticks([0,1,2], labels)
  plt.tight_layout()
  plt.show()
  plt.savefig("dept affect")
  st.image("dept affect.png")

dept_visualize()

def gender_visualize():
  st.write("**Gender Distribution**")
  label = [1,0]
  labels =[]
  for i in label:
    if i == 0:
        labels.append("Female")
    else:
        labels.append("Male")
  plt.figure(figsize=(11,5))
  sns.countplot(data = df_train, x = "Gender", hue = y , palette = "colorblind")
  plt.xticks(label, labels)
  plt.title("Effect of Employee Gender on the Attrition in the Company")
  plt.tight_layout()
  plt.show()
  plt.savefig("gender effect")
  st.image("gender effect.png")
  
gender_visualize()
st.write("**Marital Status Distribution**")
def marital_status():
  label = [1,2,0]
  labels =[]
  for i in label:
    if i == 0:
        labels.append("Divorced")
    elif i ==1:
        labels.append("Married")
    else:
        labels.append("Single")
  plt.figure(figsize=(11,6))
  sns.countplot(data = df_train, x = "MaritalStatus", hue = y , palette = "colorblind")
  plt.xticks(label, labels)
  plt.title("Effect of Employee Marital Status on the Attrition in the Company")
  plt.tight_layout()
  plt.show()
  plt.savefig("marital1 effect")
  st.image("marital1 effect.png")
  
marital_status()


def year_company():
  st.write("**Years at Company Distribution**")
  plt.figure(figsize=(11,6))
  sns.countplot(data = df_train, x = "YearsAtCompany" , hue = y , palette = "colorblind")
  plt.title("Effect of Employee Stay on the Attrition in the Company")
  plt.tight_layout()
  plt.show()
  plt.savefig("year effect")
  st.image("year effect.png")
  
year_company()
  