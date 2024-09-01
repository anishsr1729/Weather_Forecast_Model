# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#Reading CSV file and displaying it

data = pd.read_csv(r"C:\Users\anish\Desktop\pythonPractice\Dataset11-Weather-Data.csv")
print(data.head())

# Displaying shape, col, dtypes and info abt dataset

print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.info())

# Displaying the value of all elements, unique val and no of unique val of Weather column

print(data.Weather.value_counts)
print(data.Weather.unique())
print(data.Weather.nunique())

#Converting the Weather categories into Standard categories

x = 'Thunderstorms', 'Moderate Rain Showers', 'Fog'

def Create_list(x):
    list_of_lists = [w.split() for w in x]
    flat_list = list(chain(*list_of_lists))
    return flat_list

def GetWeather(list1):
    if 'Fog' in list1 and 'Rain' in list1:
        return 'RAIN+FOG'
    elif 'Snow' in list1 and 'Rain' in list1:
        return 'SNOW+RAIN'
    elif 'Snow' in list1:
        return 'SNOW'
    elif 'Rain' in list1:
        return 'RAIN'
    elif 'Fog' in list1:
        return 'FOG'
    elif 'Clear' in list1:
        return 'Clear'
    elif 'Cloudy' in list1:
        return 'Cloudy'
    else:
        return 'RAIN'
print(Create_list(x))
print(GetWeather(Create_list(x)))

# Changing the Weather col to std weather col and making all the values unique and std

data['Std_Weather'] = data['Weather'].apply(GetWeather)
print(data)
print(data.Std_Weather.value_counts())

# Sample selection and data balancing (all samples of all attr under 600)

cloudy_df = data[data['Std_Weather'] == 'Cloudy'].sample(600)
clear_df = data[data['Std_Weather'] == 'Clear'].sample(600)
rain_df = data[data['Std_Weather'] == 'RAIN']
snow_df = data[data['Std_Weather'] == 'SNOW']

# Combining the above .df's to waether.df

weather_df = pd.concat([cloudy_df, clear_df, rain_df, snow_df], axis = 0)
print(weather_df.head())    
print(weather_df.shape)
print(weather_df.Std_Weather.value_counts())

#Drop columns of Date/time and Weather

weather_df.drop(columns=['Date/Time', 'Weather'], axis=1, inplace=True)
print(weather_df.head())  
print(weather_df.shape)

#Checking duplicate variables and null variables

weather_df[weather_df.duplicated()]
print(weather_df)

weather_df.isnull().sum()
print(weather_df)
print(weather_df.dtypes)

# Describing data 
print(weather_df.describe())

#Correlations among the features

cols = ['Temp_C','Dew Point Temp_C','Rel Hum_%','Wind Speed_km/h','Visibility_km','Press_kPa',]
corr_matrix = weather_df[cols].corr()
print(corr_matrix)

# Heat Map

heat_map = sns.heatmap(corr_matrix, annot=True)
print(plt.show())

# Histogram for each feature

temp_c_hist = weather_df['Temp_C'].plot(kind='hist')
print(plt.show())
Dev_pt_temp_c_hist = weather_df['Dew Point Temp_C'].plot(kind='hist')
print(plt.show())
rel_hum_hist = weather_df['Rel Hum_%'].plot(kind='hist')
print(plt.show())
wind_speed_hist = weather_df['Wind Speed_km/h'].plot(kind='hist')
print(plt.show())
vis_hist = weather_df['Visibility_km'].plot(kind='hist')
print(plt.show())
press_hist = weather_df['Press_kPa'].plot(kind='hist')
print(plt.show())

#Converting target variable-stdweather to numeric using LabelEncoder

label_encoder = LabelEncoder()
weather_df['Std_Weather'] = label_encoder.fit_transform(weather_df['Std_Weather'])
print(label_encoder.classes_)

#Seperating independant and dependant variables

X = weather_df.drop(['Std_Weather'], axis =1)
Y = weather_df['Std_Weather']
print(X)
print(Y)

# Feature Scaling using StandardScalar

std_scalar = StandardScaler()
X_std = std_scalar.fit_transform(X)
print(X_std)

# Splitting dataset for training and testing

X_train, X_test, Y_train, Y_test,  = train_test_split(X_std, Y, test_size = 0.2, random_state =42)
print(X_train.shape)
print(X_test.shape)

# Building the model

dt_model = DecisionTreeClassifier()

# Training the model

dt_model.fit(X_train, Y_train)

# Model Predictions
Y_pred_dt = dt_model.predict(X_test)

# Model Evaluations

# Accuracy score
print('acc-sc:  ',accuracy_score(Y_test, Y_pred_dt))

# Classification Report
print('Classf-rep: ',classification_report(Y_test, Y_pred_dt))

#Confusion Matrix
print('Cnonfusion-mtrx: ',confusion_matrix(Y_test, Y_pred_dt))

# Other models
rf_model = RandomForestClassifier()
svc_model = SVC()
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression()
nb_model = GaussianNB()

# Creating a list of models and selecting the model with highest accuracy

model_list = [dt_model, rf_model, svc_model, knn_model, lr_model, nb_model]

acc_list = []
for model in model_list:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    acc_list.append(acc)

max_acc = max(acc_list)
index = acc_list.index(max_acc)
model_name = model_list[index]
print(f"Max acc is {max_acc:.2f} for {model_name}")

# K-fold cross validation

score = cross_val_score(rf_model, X_std, Y, cv = 5, scoring='accuracy')
print('Cross Validation scores: ', score)
print(score.mean())

parameters = {'n_estimators' : [50, 100], 'max_features' : ['sqrt', 'log2', None]}

grid_search = GridSearchCV(estimator=rf_model, param_grid=parameters)

grid_search.fit(X_train, Y_train)
grid_search.best_params_
print(grid_search.best_params_)
