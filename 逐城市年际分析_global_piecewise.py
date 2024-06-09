# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:45:04 2024

@author: ehlab
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font',family='Arial',size=16)
import numpy as np


year_list = [str(year) for year in range(2000,2023)]
month = 7
result = pd.DataFrame()
for year in year_list:
    data = pd.read_csv(r"H:\CE_temporal\data\night_process\para\north_"+year+"_0"+str(month)+"_night_tri.csv")
    data['ObjectID'] = data['Unnamed: 0']
    data.index = data['ObjectID']
    data = data.dropna()
    data = data[data['R2']>0.3]
    data = data[data['CE_10']>0]
    data['Year'] = year
    result = pd.concat([result,data[['Year','CE_10']]])
result['ObjectID'] = result.index
    

GUB_city = pd.read_csv(r"H:\CE_temporal\data\GUB_global_100_wgs.csv")
GUB_city.index = GUB_city['ObjectID']


from scipy.optimize import curve_fit

def sim_equation(x, a, b):
    return a * x + b 

def R2(x,y,pre):
# 计算均值
    mean_observed = np.mean(y)
# 计算总平方和（Total Sum of Squares，TSS）
    tss = np.sum((y - mean_observed)**2)
# 计算残差平方和（Residual Sum of Squares，RSS）
    rss = np.sum((y - pre)**2)
# 计算决定系数（R-squared）
    r_squared = 1 - (rss / tss)
    return r_squared


coeff = pd.DataFrame(np.nan,columns=['model','a','b','a1','b1','b2','R2_linear','piecewise_year','R2_piecewise','NAME_0','NAME_1','NAME_2'],index=[i for i in np.array(np.unique(result['ObjectID']))],dtype='float')

for i in np.array(np.unique(result['ObjectID'])):
    print(str(i)+" start")
    tmp = result[result['ObjectID']==i]
    if len(tmp) <=6:    #小于5年数据先不画图
        continue
    m = np.array(tmp['Year'],dtype=np.int16)
    n = np.array(tmp['CE_10'])
    for y in year_list:
        if y not in np.array(tmp['Year']):
            add = pd.DataFrame({'Year':[y],'CE_10':[0],'ObjectID':[i]})
            tmp = pd.concat([tmp,add])
    

    tmp['Year'] = pd.Categorical(tmp['Year'], categories=year_list, ordered=True)
    tmp = tmp.sort_values('Year')
    tmp.Year = tmp.Year.astype(int)
    
    tmp_r2 = []
    tmp_a1 = []
    tmp_b1 = []
    tmp_b2 = []
    
    for year in m[3:-3]:
        def piecewise_equation(t,a1,b1,b2): 
            return np.piecewise(t, [t <= year, t > year], [lambda t:a1*t+b1, lambda t:(a1-(b2-b1)/year)*t+b2])
        
        params, covariance = curve_fit(piecewise_equation,m,n)
        # params, covariance = curve_fit(lambda m, a,b,c,d: piecewise_equation(m, a,b,c, year), m, n)

        a1,b1,b2 = params
        tmp_a1.append(a1)
        tmp_b1.append(b1)
        tmp_b2.append(b2)
        tmp_r2.append(R2(m,n,piecewise_equation(m,a1,b1,b2)))
    
    params, covariance = curve_fit(sim_equation,m,n)
    a, b = params
    r2_linear = R2(m,n,sim_equation(m,a,b))
    
    
    plt.figure(figsize=[14,5])
    ax = plt.gca()
    m_inter=(2022-2000)/100
    m_fake = [2000 + m_f*m_inter for m_f in range(101)]
    plt.scatter(m,n,color='black')
    # plt.plot(m_fake,sim_equation(np.array(m_fake), a,b),color='r')
    plt.ylabel("CE_10")
    plt.xticks([x for x in range(2000,2023)])
    plt.xlabel("Year")
    if isinstance(GUB_city['NAME_2'][i],float)==False:
        plt.title("Month-"+str(month)+" " +GUB_city['NAME_0'][i]+"-"+GUB_city['NAME_1'][i]+"-"+GUB_city['NAME_2'][i])
    else:
        plt.title("Month-"+str(month)+" "+GUB_city['NAME_0'][i]+"-"+GUB_city['NAME_1'][i])
    
    
    
    coeff['NAME_0'][i] = GUB_city['NAME_0'][i]
    coeff['NAME_1'][i] = GUB_city['NAME_1'][i]
    coeff['NAME_2'][i] = GUB_city['NAME_2'][i]
    
    
    if r2_linear>np.nanmax(tmp_r2):
        plt.plot(m_fake,sim_equation(np.array(m_fake), a,b),color='r')
        coeff['model'][i] = 'linear'
        
    else:
        year = m[3:-3][np.where(tmp_r2 == np.nanmax(tmp_r2))[0][0]]
        def piecewise_equation(t,a1,b1,b2): 
            return np.piecewise(t, [t <= year, t > year], [lambda t:a1*t+b1, lambda t:(a1-(b2-b1)/year)*t+b2])
        
        a1 = tmp_a1[np.where(tmp_r2 == np.nanmax(tmp_r2))[0][0]]
        b1 = tmp_b1[np.where(tmp_r2 == np.nanmax(tmp_r2))[0][0]]
        b2 = tmp_b2[np.where(tmp_r2 == np.nanmax(tmp_r2))[0][0]]
        coeff['model'][i] = 'piecewise'
        coeff['a1'][i] = a1
        coeff['b1'][i] = b1
        coeff['b2'][i] = b2
        coeff['piecewise_year'][i] = year
        plt.plot(m_fake,piecewise_equation(np.array(m_fake), a1,b1,b2),color='r')
    
    coeff['a'][i] = a
    coeff['b'][i] = b
    coeff['a1'][i] = a1
    coeff['b1'][i] = b1
    coeff['b2'][i] = b2
    ax.text(0.8,0.2,"R2="+str(round(np.max([r2_linear,np.nanmax(tmp_r2)]),2)),va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),),size=16, transform = ax.transAxes)
    coeff['R2_linear'][i] = r2_linear
    coeff['R2_piecewise'][i] = np.nanmax(tmp_r2)
    
    plt.tight_layout()
    if os.path.exists(r"H:\CE_temporal\data\night_process\temporal_figs")==False:
        os.makedirs(r"H:\CE_temporal\data\night_process\temporal_figs")
    plt.savefig(r"H:\CE_temporal\data\night_process\temporal_figs\\"+str(i)+".tif",dpi=300)

coeff['ObjectID'] = coeff.index
coeff.to_csv(r"H:\CE_temporal\data\night_process\temporal_para\\07.csv",index=False)