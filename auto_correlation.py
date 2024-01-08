"""
Copyright (C) 2023
@ Name: auto_correlation.py
@ Time: 2023/12/20 21:36
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import kpss,acf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr,pearsonr
data = pd.read_csv('C:\Study\Code\数据集\PP_GAS\gt_2011.csv',header=0)
name = data.columns



data = data.values



print(1)
# 计算自相关系数
fig1, ax1 = plt.subplots(3,4,layout='constrained')

for i in range(data.shape[1]):
    ax = ax1.flat[i]
    acf_result = acf(x=data[:, i], fft=True,nlags=30)
    diff = np.diff(data[:, i])
    dff_result = acf(x=diff, fft=True,nlags=30)
    ax.bar(x=np.arange(acf_result.shape[0]) + 1, height=acf_result, label=name[i])
    ax.bar(x=np.arange(dff_result.shape[0]) + 1, height=dff_result, label="Diff".format(i + 1))
    ax.legend()

scaler = MinMaxScaler()
pv_data = scaler.fit_transform(data)
uns_index = []


fig2, ax2 = plt.subplots(3,4,layout='constrained')

for i in range(data.shape[1]):
    ax = ax2.flat[i]
    kpsstest = kpss(pv_data[:, i], regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    if kpss_output["p-value"] > 0.05:
        print("Kpss Test V{} stationary: ".format(i + 1), kpss_output["p-value"])

        ax.plot(pv_data[:, i], label="{} stationary".format(name[i]))
        ax.legend()
    else:
        print("Kpss Test V{} unstationary: ".format(i + 1), kpss_output["p-value"])
        ax.plot(pv_data[:, i], label="{} unstationary".format(name[i]))
        uns_index.append(i)
        ax.legend()



# spearman 热力图
pearsonr_matrix = np.zeros(shape=(pv_data.shape[1],pv_data.shape[1]))
spearmanr_matrix = np.zeros(shape=(pv_data.shape[1],pv_data.shape[1]))

for i in range(pv_data.shape[1]):
    for j in range(pv_data.shape[1]):
        pearsonr_matrix[i][j] = pearsonr(pv_data[:,i],pv_data[:,j])[0]
        spearmanr_matrix[i][j] = spearmanr(pv_data[:,i],pv_data[:,j])[0]

if data.shape[1] % 2 != 0:
    plt.delaxes(ax1[-1, -1])
    plt.delaxes(ax2[-1,-1])

plt.figure()
sns.heatmap(pearsonr_matrix,fmt='.2g',annot=True)

plt.figure()
sns.heatmap(spearmanr_matrix,annot=True)
plt.show()
