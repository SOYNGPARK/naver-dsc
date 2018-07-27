# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:44:54 2018

@author: soug9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(r'C:\Users\soug9\Desktop\occupancy_data\datatraining.txt')
test = pd.read_csv(r'C:\Users\soug9\Desktop\occupancy_data\datatest.txt')

data = pd.concat([train, test])

data['datetime'] = pd.to_datetime(data['date'])
data = data.sort_values('datetime')
data.index = range(len(data))
data.columns


# 전처리
data.boxplot(column='Light', by='Occupancy')

data[(data['Occupancy']==1) & (data['Light']>=1000)].index
data[['Light', 'Occupancy']][(data['Occupancy']==1) & (data['Light']>=1000)]

data[(data['Occupancy']==0) & (data['Light']>=800)].index
data[['Light', 'Occupancy']][(data['Occupancy']==0) & (data['Light']>=800)]

# Occupancy==1 일 때, 2601, 2602, 2603
plt.plot(data['Light'][2550:2650])
data['Light'][2600:2605]
x = data['Light'].loc[[2600, 2604]]
data['Light'][2601:2604] = [np.percentile(x, 75), np.percentile(x, 50), np.percentile(x, 25)]

# Occupancy==0 일 때, 6496, 6497, 6498
plt.plot(data['Light'][6000:7000])
data['Light'][6495:6500]
x = data['Light'].loc[[6495, 6499]]
data['Light'][6496:6499] = [np.percentile(x, 75), np.percentile(x, 50), np.percentile(x, 25)]


# 파생변수

# 파생 변수 1 : 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio' 변화량
# 1-1) 변화량 = 이전 값과의 차
def difference_from_prior(feature) :
    dif = [feature[i] - feature[(i-1)] for i in range(1, len(feature))]
    dif = [None] + dif
    return dif

data['TemperatureDif'] = difference_from_prior(data['Temperature'])
data['HumidityDif'] = difference_from_prior(data['Humidity'])
data['LightDif'] = difference_from_prior(data['Light'])
data['CO2Dif'] = difference_from_prior(data['CO2'])
data['HumidityRatioDif'] = difference_from_prior(data['HumidityRatio'])
   
# 1-2) 변화량 = 이전 값 n개의 평균보다 크면 1(증가), 작으면 0(감소)
def increase(feature, n=10) :
    incr = list(map(lambda i : 1 if feature[i] > np.mean(feature[(i-n):i]) else 0, range(n, len(feature))))
    incr = [None]*n + incr
    return incr

data['TemperatureIncr'] = increase(data['Temperature'])
data['HumidityIncr'] = increase(data['Humidity'])
data['LightIncr'] = increase(data['Light'])
data['CO2Incr'] = increase(data['CO2'])
data['HumidityRatioIncr'] = increase(data['HumidityRatio'])

# 파생 변수 2 : 'date' -> categorical 변수
# 2-1) 주중(0), 주말(1) - 0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
data['weekend'] = list(map(lambda x : 1 if x in [5, 6] else 0 ,data['datetime'].dt.weekday))

# 2-2) non-office hour(0), office hour(1) - 7 to 19
data['office'] = list(map())



























