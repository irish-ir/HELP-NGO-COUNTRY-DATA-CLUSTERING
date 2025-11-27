import streamlit as st 
import numpy as np
import pandas as pd
import joblib

# Let's load the joblib instances over here
with open('pipeline.joblib','rb') as file:
    preprocess=joblib.load(file)

with open('model.joblib','rb') as file:
    model= joblib.load(file) 

# Let's take the input 
st.title('HELP NGO ORGANIZATION')   
st.subheader('THIS WILL HELP TO IDENTIFY THE DEVELOPMEMT CATEGORY OF THE COUNTRY USING SOCIO-ECONOMIC FACTORS. THE OROIGINAL DATA HAS BEEN CLUSTERED USING KMEANS. ') 

# Let's take the inputs
gdpp = st.number_input('Enter the GDPP of a country (GDP per population)')
income = st.number_input('Enter income per population')
imports = st.number_input('Imports of goods and services per capita. Given as %age of the GDP per capita')
exports = st.number_input('Exports of goods and services per capita. Given as %age of the GDP per capita')
inflation = st.number_input('The measurement of the annual growth rate of the Total GDP')
life_expcy = st.number_input('The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert = st.number_input('The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health = st.number_input('Total health spending per capita. Given as age of GDP per capita')
child_mort = st.number_input('Death of children under 5 years of age per 1000 live births')

input_list=[child_mort,health,fert,life_expcy,inflation,exports,imports,income,gdpp]

final_input_list=preprocess.transform([input_list])
if st.button('Predict'):
    prediction=model.predict(final_input_list)[0]
    if prediction==0:
        st.success('Developing')
    elif prediction == 1:
        st.success('Developed')
    else:
        st.error('Under Developed')        
