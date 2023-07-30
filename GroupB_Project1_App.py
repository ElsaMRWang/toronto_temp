import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

header = st.container()
dataset = st.container()


with header:
  st.title("Predict monthly average temperature in Toronto")
  st.text('This project focuses on predicting Toronto monthly average temperature (C) using ARIMA model and data since January 2010')


with dataset:
  st.header('1.Monthly Average Temperature in Toronto')
  st.text('Data collected from Data Download for Toronto')

  trt_temp = pd.read_csv("trt_temp.csv")
  trt_temp = pd.DataFrame( trt_temp)  
  trt_temp['date']= pd.to_datetime(trt_temp['date'])
  trt_temp = trt_temp[(trt_temp['date'] >= '2010-01-01') & (trt_temp['date'] <= '2023-06-30')]
  trt_temp = trt_temp[['date', 'avg_temperature']]
  trt_temp.rename({'avg_temperature': 'temp'}, axis=1, inplace=True)
  trt_temp.set_index('date', inplace = True)
  trt_temp=trt_temp['temp'].resample('M').mean()
  trt_temp = pd.DataFrame(trt_temp)

  st.line_chart(trt_temp)
   
  st.header('2.Raw Data Decomposition')
  decomposition = sm.tsa.seasonal_decompose(trt_temp, model = 'additive')
  fig = decomposition.plot()
  st.pyplot(fig)

  adftest = adfuller(trt_temp)
  trt_temp1=trt_temp.diff(1)
  adftest1 = adfuller(trt_temp1.dropna())
  
  st.header('3.Dickey-Fuller Test & Auto Arima Model Building')
  st.subheader("3-1.Zero Difference Dickey-Fuller Test: P-value is")
  st.write(adftest[1])

  st.subheader("3-2.First Difference Dickey-Fuller Test: P-value is")
  st.write(adftest1[1])

  len(trt_temp)
  train=trt_temp[:140]
  test=trt_temp[140:]

  arima_model = auto_arima(train, start_p=0, d=1, start_q=0,
    max_p=5, max_d=1, max_q=5, start_P=0,
    D=1, start_Q=0, max_P=5, max_D=1,
    max_Q=5, m=12, seasonal=True,
    error_action='warn', trace = True,
    supress_warnings=True, stepwise = True,
    random_state=20,n_fits = 50 )

  st.header('4.Arima Model Summary: p=0, d=1, q=2')
  st.write(arima_model.summary())


  st.subheader('4-1.Model Performance Using Test Dataset')
  st.subheader("Prediction Table")

  test_prediction = pd.DataFrame(arima_model.predict(n_periods = len(test)), index=test.index)
  test_prediction.columns = ['prediction']
  test_prediction

  fig_1 = plt.figure(figsize=(8,5))
  plt.plot(train, label="Train")
  plt.plot(test, label="Test")
  plt.plot (test_prediction, label="Prediction")
  plt.legend()
  plt.title("Predictions Using Test Dataset")
  st.pyplot(fig_1)
  
  st.subheader("4-2.R-square of this model is:")
  R_square = r2_score(test, test_prediction)
  st.write(R_square)

  st.subheader("4-3.Root of Mean Squared Error of this model is:")
  RMSE = np.sqrt(mean_squared_error(test, test_prediction))
  st.write(RMSE)


  st.subheader("4-4.Future Predictions")
  future_prediction = arima_model.predict(100)
  future_prediction.columns = ['prediction']
  future_prediction 

  fig_2 = plt.figure(figsize=(8,5))
  plt.plot(trt_temp, label="Actual")
  plt.plot (future_prediction, label="Prediction")
  plt.legend()
  plt.title("Predictions Up Until December 2029")
  st.pyplot(fig_2)



 


