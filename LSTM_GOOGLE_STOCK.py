# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:20:04 2020

@author: begas
"""

# -------------------------------------------------------------------
# 파일명 : LSTM_GOOGGLE_STOCK.py
# 설  명 : 파이썬을 활용한 LSTM
# 작성자 : 이재병(010-2775-0930, jblee@begas.co.kr)
# 작성일 : 2020/03/27
# 패키지 : numpy, pandas ......
# LSTM 참고자료 : https://www.youtube.com/watch?v=arydWPLDnEc
# 데이터 다운로드 : https://finance.yahoo.com/quote/GOOG/history/
# --------------------------------------------------------------------
#%% 텐서플로우 주의사항
# 주 의 : 
# 1. Tensorflow 설치되어 있어야 함(보통 가상환경에 설치)
# []안에 글은 cmd 창에 쳐야함
# 가상환경 생성 : [conda create -n tf]
# 가상환경 삭제 : [conda env remove -n tf]
# 텐서플로우 설치 : [conda activate tf] - [conda install tensorflow]
# Spyder 재설치 해야함 : [conda install -n tf spyder]

#%% 패키지 불러오기
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#작업공간 설정
import os
os.getcwd()
os.chdir("C:/Pyproject")

#Padas display option
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 80)

#%%Seed 설정 (중요)
tf.random.set_seed(2019)

#%% LSTM을 위한 데이터 준비하기

#데이타 불러오기
data = pd.read_csv("./DAT/GOOG.csv", date_parser = True)
data.tail()
data.info()

#학습 테스트 데이터 나누기
data_train = data[data["Date"]<"2019-01-01"].copy()
data_test = data[data["Date"]>="2019-01-01"].copy()

#필요없는 변수 제거하기
training_data = data_train.drop(["Date","Adj Close"], axis = 1)
test_data = data_test.drop(["Date","Adj Close"], axis = 1)

#Train 데이터 스케일링하기
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

#학습, 테스트 데이터 형태로 바꾸기(1:60일 데이터로 60일째의 y예측)
x_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i,0])

#array로 설정
x_train, y_train = np.array(x_train), np.array(y_train)
print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)

#%% LSTM 모델 구축
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


#LSTM 모델
regressior = Sequential()
#Layer 1
regressior.add(LSTM(units = 60, activation = "relu", return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
regressior.add(Dropout(0.2))
#Layer 2
regressior.add(LSTM(units = 60, activation = "relu", return_sequences = True))
regressior.add(Dropout(0.2))
#Layer 3
regressior.add(LSTM(units = 80, activation = "relu", return_sequences = True))
regressior.add(Dropout(0.2))
#Layer 4
regressior.add(LSTM(units = 120, activation = "relu"))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.summary()


#Compile
regressior.compile(optimizer = "adam", loss = "mean_squared_error")

#Model fit
regressior.fit(x_train, y_train, epochs = 10, batch_size = 32)

#%% 테스트 데이터셋 준비

#LSTM에서 첫 번째 테스트 y를 예측하기 위해서는 train의 마지막 60일치 데이터 필요
past_60_days = data_train.tail(60)
#따라서 train의 마지막 60일치 데이터를 test데이터와 병합
df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(["Date","Adj Close"], axis = 1)
df.head()

#테스트 데이터 스케일링
#Train에서는 .fit_transform했는데 Test는 transform만 하는 이유는
#Train에서 fit으로 데이터를 변형한 형태를 Test에 동일하게 적용해야하기 때문에 다시 fit 하지 않음
#ex) Train에서 fit으로 x-10만큼 변화시켰으면 Test에도 동일하게 x-10만큼 변화시키기위해 transform만 사용
#fit을 또 하면 test데이터에 맞게 x-20등으로 변환 되어버림
inputs = scaler.transform(df)


#학습, 테스트 데이터 형태로 바꾸기(1:60일 데이터로 60일째의 y예측)
x_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
print("x_test shape ",x_test.shape)
print("y_test shape ",y_test.shape)

#예측하기
y_pred = regressior.predict(x_test)

#Scale 다시 돌려주기
print(scaler.scale_)
scale = 1/scaler.scale_[0] #Y의 스케일 다시 돌려주기

y_pred = y_pred * scale
y_test = y_test * scale

#%%예측결과 시각화 하기
plt.figure(figsize=(14,5))
plt.plot(y_test, color = "red", label = "Real Google Stock Price")
plt.plot(y_pred, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
plt.savefig("./OUT/GOOG_STOCK_Prediction.png", dpi=300)

#%% 학습한 모델 저장하고 불러오기
from tensorflow.keras.models import load_model
#MODEL 저장
regressior.save('./MODEL/GOOG_LSTM.h5')
# #저장한 모델 불러오기
# LSTM_MODEL = load_model('./MODEL/GOOG_LSTM.h5')
# LSTM_MODEL.predict(x_test)