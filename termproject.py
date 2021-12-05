import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('data.csv')

df.groupby(['regionalHQ'])[['dispatch', 'transport', 'patient']].mean() #지역별 출동건수, 이송건수, 이송환자수

df.groupby('month')[['dispatch', 'transport', 'patient']].mean()

dataset1 = pd.DataFrame({'dispatch':df.groupby('month')['dispatch'].mean()})
dataset2 = pd.DataFrame({'dispatch':df['dispatch'], 'patient':df['patient']})

fig = plt.figure()
axes1 = fig.add_subplot(2, 1, 1)
axes2 = fig.add_subplot(2, 1, 2)
axes1.plot(dataset1['dispatch'])
axes2.plot(dataset2['dispatch'], dataset2['patient'], 'o')
axes1.set_title("month-dispatch") #월별 출동건수
axes2.set_title("dispatch-patient") #출동건수당 이송환자수

dis_arr = np.array(dataset2['dispatch'].values).reshape(-1,1)
pat_arr = np.array(dataset2['patient'].values).reshape(-1,1)

lr = LinearRegression()
lr.fit(dis_arr, pat_arr)
w = lr.coef_[0]
print(w) #출동건수당 이송환자수의 기울기
plt.scatter(dis_arr, pat_arr) #실제 데이터 그래프
plt.plot(dis_arr, dis_arr * w, c = 'red') #추세선 그래프

ft = df['regionalHQ'].isin(['chungbuk']) #충북 내의 데이터를 test로 활용
dataset3 = pd.DataFrame({'dispatch':df.loc[ft, 'dispatch'], 'patient':df.loc[ft, 'patient']})
dis_test = np.array(dataset3['dispatch'].values).reshape(-1,1)
pat_test = np.array(dataset3['patient'].values).reshape(-1,1)
pat_test_pred = lr.predict(dis_test)
test_loss = mean_squared_error(pat_test, pat_test_pred)
pat_pred = lr.predict(dis_arr)
train_loss = mean_squared_error(pat_arr, pat_pred)
print(test_loss, train_loss)
plt.scatter(dis_arr, pat_arr) #실제 데이터 그래프
plt.plot(dis_arr, dis_arr * w, c = 'red') #추세선 그래프
plt.scatter(dis_test, pat_test_pred, c = 'purple') #테스트 예측 그래프
r2 = r2_score(pat_test, pat_test_pred)
print(r2)
