import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.models import load_model

with open('preprocessor.pkl','rb') as f:
    preprocessor = pickle.load(f)
model = load_model(filepath='best_model.h5')


st.title('Customer Churn Prediction')

CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography',['France','Germany','Spain'])
Gender =  st.selectbox('Gender',['Female','Male'])
Age = st.slider('Age',18,92)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Num Of Products',0,4)
HasCrCard = st.selectbox('Has Credit Card',[0,1])
IsActiveMember = st.selectbox('Is Active Member',[0,1])
EstimatedSalary = st.number_input('Estimated Salary')

input = pd.DataFrame(
    data={
    'CreditScore' : [CreditScore],
    'Geography' : [Geography],
    'Gender':[Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
    }
)

input_processed = preprocessor.transform(input)
prediction = model.predict(input_processed)
probability = prediction[0][0]
st.write(f'Prediction for churn is {probability}')
if probability>0.5:
    st.write('Customer will Churn')
else:

    st.write('Customer will not Churn')

