import streamlit as st
import numpy as np
import pandas as pd
import xlrd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

st.title('EMPLOYEE PERFORMANCE EVALUATION SYSTEM')
st.header('Employee performance Evaluation')
st.write('Employee performance evaluation is a critical aspect of every company as it ensures that every employee contributes to the goals and objectives of the organization.This application enables the company monitor employee perfomance through machine learning')

df=pd.read_excel("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xls")

#display data
st.subheader('sample data')
st.table(df.head())

#Data Processing

#create feature and target data
X=df.drop('PerformanceRating',axis=1)
y=df['PerformanceRating']


#Encode data
cat_columns = df.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder()
df[cat_columns] = encoder.fit_transform(df[cat_columns])

#Scale Data
scaler=StandardScaler()
scaled_df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)


#Select columns to fit the model
columns= ['EmpEnvironmentSatisfaction','EmpLastSalaryHikePercent','ExperienceYearsInCurrentRole','YearsSinceLastPromotion']
X= X[columns]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Get user inputs
name=st.text_input('What is your name?').capitalize()

#Get feature input from user
if name!="":
    st.write('Hello{} Please complete the form below'.format(name))
else:
    st.write('Please enter your name')

import streamlit as st
import pandas as pd

def get_user_input():
    EmpEnvironmentSatisfaction = st.sidebar.slider('EmpEnvironmentSatisfaction', 1, 4)
    EmpLastSalaryHikePercent = st.sidebar.slider('EmpLastSalaryHikePercent', 11, 25)
    ExperienceYearsInCurrentRole = st.sidebar.slider('ExperienceYearsInCurrentRole', 0, 18)
    YearsSinceLastPromotion = st.sidebar.slider('YearsSinceLastPromotion', 0, 16)

    # Store the user data into a dictionary
    user_data = {
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion
    }

    # Transform the data into a DataFrame
    user_input_df = pd.DataFrame(user_data, index=[0])
    return user_input_df

# Get user input
user_input_df = get_user_input()

# Display user input
st.subheader('Below is the user input')
st.dataframe(user_input_df)

#create a button to ask user to get results
bt= st.button('Get Result')


if bt:
   rf=RandomForestClassifier(random_state=1)
   rf.fit(X_train,y_train)

   
   prediction=rf.predict(user_input_df)

   if-prediction==1:
        
        st.write('Hello,you have performed well')
   else:
        st.write('Hello, your performance needs improvement')

    #Display model score
        st.write('Model Accuracy: ', metrics.accuracy_score(y_test,rf.predict(X_test)))





                                                   


