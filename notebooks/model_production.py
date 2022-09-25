import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier #boosting


#load dataset
df = pd.read_csv('../data/bank-additional-full.csv',sep = ';')

#data preprocesing
df=pd.get_dummies(data=df, drop_first=True)

#feature split in indepent and target
X = df.loc[:,df.columns != 'y_yes']
y = df.loc[:,'y_yes']

#split train and test group's
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,\
                                random_state=12)

#StandardScaler fit and trasnform
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)