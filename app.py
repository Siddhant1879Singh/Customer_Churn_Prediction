import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

## Load the trained model,scaler pickle ,onehot
model=tf.keras.models.load_model('model.h5')

## Load the encoder and scaler
with open('ohe_geography.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('label_gender_encoder.pkl','rb') as file:
    gender_encoder=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
## Streamlit app
st.title("Customer Churn Prediction")

## User input
geography=st.selectbox('Geography',geo_encoder.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is active member',[0,1])

## Input data preparation
input_data={
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':gender,
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}

## Now we have to convert this into the input for model
df=pd.DataFrame.from_dict([input_data])

# One hot encode 'Geography'
geo_encoded=geo_encoder.transform(df[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out(['Geography']))
df=pd.concat([df.drop('Geography',axis=1),geo_encoded_df],axis=1)
df['Gender']=gender_encoder.transform(df['Gender'])

## Scaling the input data
input_scaled=scaler.transform(df)

prediction=model.predict(input_scaled)

prediction_proba=r=prediction[0][0]
st.write(f"The churn probability is: {prediction_proba}")
if prediction_proba>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")