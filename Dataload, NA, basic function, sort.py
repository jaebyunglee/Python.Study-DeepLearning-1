# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:26:36 2020

@author: begas
"""

# -------------------------------------------------------------------
# 파일명 : 데이터불러오기,NA,함수,정렬.py
# 설  명 : 파이썬 기본(데이터 불러오기, NA확인, 기본함수 적용, 정렬 등)
# 작성자 : 이재병(010-2775-0930, jblee@begas.co.kr)
# 작성일 : 2020/03/20
# 패키지 : pandas
# --------------------------------------------------------------------

#%% 필요 패키지 불러오기 
import pandas as pd 

#%% 작업공간 확인 및 변경 
import os
os.getcwd() #현재 작업공간 확인
os.chdir('C:/Pyproject') #작업공간 변경

#%% 데이터 불러오기
train = pd.read_csv('./DAT/train.csv')
test = pd.read_csv('./DAT/test.csv')

#%% 데이터 살펴보기
train.shape #행과 열의 개수
train.columns #변수 명 살펴보기
train.head(5)
train.info()
train.describe() #요약통계량 살펴보기

#Padas display option
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 80)

#NA 개수 확인하기
train.isnull().sum()
train.isnull().sum()/train.shape[0]

#%% 데이터 인덱싱 하기
#열
train["PoolQC"] #하나의 열 가져오기
train[["PoolQC", "Fence"]] #두개 이상의 열 가져오기
train["AA"] = 0 # 열 추가하기
train.drop("PoolQC",1) # PoolQC 열 제거

#행
train[0:3] # 행 추출
train[train["MiscVal"]!=0] # 특정 조건을 만족하는 행 추출
train.drop(range(0,5),0) # 1~5 행 제거

#loc 행과 열의 이름으로 인덱싱
train.loc[0:5,["PoolQC","Fence"]]
train.loc[[0,1,5],["PoolQC","Fence"]]

#iloc 행과 열의 인덱스로 인덱싱
train.iloc[0:5,1:3]
train.iloc[[0,1,5],[1,2]]

#Boolean 인덱싱 하기
train.loc[-train["PoolQC"].isnull(),] #PoolQC가 NaN이 아닌 행만 추출


#%% 결측값 다루기
# NA 제거하고 인덱싱하기
train.dropna(how="any") # NA가 하나라도 있는 행 제외하고 인덱싱
train.dropna(how="all") # 전부 NA인 행 제외하고 인덱싱
train[train.isnull().sum(axis=1)!=0] # NA가 하나라도 있는 행 추출
train[-train.isnull()["PoolQC"]] #PoolQC가 NA가 아닌 행만 추출

# NA 채워넣기
train["PoolQC"].fillna(value="N") # PoolQC 열의 NA를 "N"으로 대체


#%% 기본함수 적용하기
train.sum(axis=0)
train.sum(axis=1)
train["SalePrice"].sum() #합계, 행(axis=1, skipna=True)
train["SalePrice"].median() #중위수
train["SalePrice"].mean() #평균, 행(axis=1, skipna=True)
train["SalePrice"].var() #분산
train[["SalePrice","OverallQual"]].cov() # 공분산
train[["SalePrice","OverallQual"]].corr() # 상관계수
train["SalePrice"].count() # NA가 아닌 값의 개수
train["SalePrice"].max() # 최대값
train["SalePrice"].argmax() #최대값의 위치

#%% 데이터 정렬하기
train.sort_index(axis=0, ascending=True) # 행인덱스 오름차순
train.sort_index(axis=1, ascending=True) # 열인덱스 오름차순
train.sort_values(by=["SalePrice","OverallQual"], ascending=True) # 두 열기준 오름차순
