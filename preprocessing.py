# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:44:54 2018

@author: soug9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data

train = pd.read_csv(r'C:\Users\soug9\Desktop\occupancy_data\original\datatraining.txt')
test1 = pd.read_csv(r'C:\Users\soug9\Desktop\occupancy_data\original\datatest.txt')
test2 = pd.read_csv(r'C:\Users\soug9\Desktop\occupancy_data\original\datatest2.txt')

data = pd.concat([train, test1, test2])

data['date'].head()

data['datetime'] = pd.to_datetime(data['date'])
data = data.sort_values('datetime')
data.index = range(len(data)) # index 초기화

data.columns
data['datetime'].dt.day.value_counts().sort_index()


# 전처리 -Light
ax = plt.gca()
ax.set_ylim([0, 1750])
plt.plot(data['Light'])

data.boxplot(column='Light', by='Occupancy')

data[(data['Occupancy']==1) & (data['Light']>=1000)].index
data[['Light', 'Occupancy']][(data['Occupancy']==1) & (data['Light']>=1000)]

data[(data['Occupancy']==0) & (data['Light']>=800)].index
data[['Light', 'Occupancy']][(data['Occupancy']==0) & (data['Light']>=800)]

# Occupancy==1 일 때, 2601, 2602, 2603, 2604
plt.plot(data['Light'][2590:2620])
data['Light'][2595:2606]
x = data['Light'].loc[[2599, 2605]]
data['Light'][2600:2605] = [np.percentile(x, 16), np.percentile(x, 33), np.percentile(x, 50), np.percentile(x, 67), np.percentile(x, 84)]

# Occupancy==1 일 때, 11946, 11947, 11948
plt.plot(data['Light'][11000:13000])
data['Light'][11945:11950]
x = data['Light'].loc[[11945, 11949]]
data['Light'][11946:11949] = [np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75)]

# Occupancy==1 일 때, 13389
plt.plot(data['Light'][12900:13800])
data['Light'][13388:13391]
x = data['Light'].loc[[13388, 13390]]
data['Light'][13389] = np.percentile(x, 50)

# Occupancy==0 일 때, 6495, 6496, 6497, 6498
plt.plot(data['Light'][6000:7000])
data['Light'][6494:6500]
x = data['Light'].loc[[6494, 6499]]
data['Light'][6495:6499] = [np.percentile(x, 20), np.percentile(x, 40), np.percentile(x, 60), np.percentile(x, 80)]

# 파생변수 

# 파생 변수 1 : 'Temperature', 'Humidity', 'CO2', 'HumidityRatio' 변화량
# 각 데이터 셋에 시간 차가 있으므로 train, test1, test2 따로 실행해야함

# 1-1) 변화량 = 이전 값과의 차
def difference_from_prior(feature) :
    dif = [feature[i] - feature[(i-1)] for i in range(1, len(feature))]
    dif = [None] + dif
    return dif

train['TemperatureDif'] = difference_from_prior(train['Temperature'])
train['HumidityDif'] = difference_from_prior(train['Humidity'])
train['CO2Dif'] = difference_from_prior(train['CO2'])
train['HumidityRatioDif'] = difference_from_prior(train['HumidityRatio'])
   
# 1-2) 변화량 = 이전 값 n개의 평균
def increase(feature, n=10) :
    incr = list(map(lambda i : np.mean(feature[(i-n):i]), range(n, len(feature))))
    incr = [None]*n + incr
    return incr

train['TemperatureIncr'] = increase(train['Temperature'])
train['HumidityIncr'] = increase(train['Humidity'])
train['CO2Incr'] = increase(train['CO2'])
train['HumidityRatioIncr'] = increase(train['HumidityRatio'])

# 파생 변수 2 : 'date' -> categorical 변수
# 2-1) 주말(0), 주중(1) - 0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
data['weekday'] = list(map(lambda x : 0 if x in [5, 6] else 1 ,data['datetime'].dt.weekday))

# 2-2) non-office hour(0), office hour(1) - 7 to 19
data['office'] = list(map(lambda x: 1 if 7 <= x <= 19 else 0, data['datetime'].dt.hour))
data['office'][data['weekday']==0] = 0

# 2-3) 새벽(0-6)(0), 오전(6-12)(1), 오후(12-18)(2), 밤(18-24)(3)
data['timegroup'] = [0]*len(data)
data['timegroup'][(6<=data['datetime'].dt.hour) & (data['datetime'].dt.hour<12)] = 1
data['timegroup'][(12<=data['datetime'].dt.hour) & (data['datetime'].dt.hour<18)] = 2
data['timegroup'][18<=data['datetime'].dt.hour] = 3

























