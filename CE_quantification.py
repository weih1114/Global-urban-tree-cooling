# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:37:04 2024

@author: WH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rc('font',family='Arial',size=16)

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

for inte in range(1,354):
    if os.path.exists(r"F:\CE_temporal\data\LST_grid\PTree_2015\\"+str(inte)+".csv") == False:
        continue
    if os.path.exists(r"F:\CE_temporal\data\LST_grid\LST_day\\"+str(inte)+".csv")== False:
        continue
    ptree = pd.read_csv(r"F:\CE_temporal\data\LST_grid\PTree_2015\\"+str(inte)+".csv")
    ptree.index = ptree['gridID']
    lst = pd.read_csv(r"F:\CE_temporal\data\LST_grid\LST_day\\"+str(inte)+".csv")
    lst.index = lst['gridID']
    
    data = pd.merge(lst,ptree, left_index=True, right_index=True)
    data.replace(-9999, np.nan, inplace=True)

    date_list = data.columns[1:-7]
    print(str(inte)+" start")
    for d in date_list:
        tmp = data[[d,'land_cover_2015']]
        tmp = tmp.dropna()
        tmp[d] = tmp[d]*0.02
        tmp['land_cover_2015'] = tmp['land_cover_2015']/784
    
        #去掉相同的值
        x = []
        y = []
        ptree_list = np.array(np.unique(tmp['land_cover_2015']))
        for ptree in ptree_list:
            x.append(ptree)
            y.append(np.nanmean(tmp[d][tmp['land_cover_2015'] == ptree]))
    
        data_post = pd.DataFrame(columns=['lst','ptree'])
        data_post['lst'] = y
        data_post['ptree'] = x
    
        #20个点合在一起
        x = []
        y = []
        tmp = data_post.sort_values('ptree',inplace = False)
        number = int(len(tmp)/20)
        for i in range(number):
            group = pd.DataFrame(0,columns=['lst','ptree'],index=tmp.index[i*20:(i+1)*20],dtype='float')
            for j in group.index:
                group['ptree'][j] = tmp['ptree'][j]
                group['lst'][j] = tmp['lst'][j]
            FTC = group['ptree'][(group['lst'].isna()==False) & (group['lst'] >= 5)] #第d天每个组要超过5个数据
            LST = group['lst'][(group['lst'].isna()==False) & (group['lst'] >= 5)]
            x.append(np.nanmean(FTC))
            y.append(np.nanmean(LST))
    
        # if len(x)<20:
        #     continue
        x = np.array(x)
        y = np.array(y)
        params, covariance = curve_fit(sim_equation, x, y)
        a, b = params
    
        fig,ax = plt.subplots(figsize=(6,4))
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        plt.scatter(x,y)
        plt.plot(x,sim_equation(x,a,b),color='r')

        plt.xlabel("tree")
        plt.ylabel('lst')
        plt.title(d)
    
        ax = plt.gca()
    # plt.text(0.1,0.2, "text", fontsize = 20, transform = ax.transAxes)
        ax.text(0.8,0.7,"R2="+str(round(R2(x,y,sim_equation(x,a,b)),2)),va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),),size=16, transform = ax.transAxes)
        ax.text(0.8,0.9,"SS="+str(len(x)),va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),),size=16,transform = ax.transAxes)

        plt.tight_layout()
        if os.path.exists(r"F:\CE_temporal\data\process\figs\\"+str(inte)) == False:
            os.mkdir(r"F:\CE_temporal\data\process\figs\\"+str(inte))
        plt.savefig(r"F:\CE_temporal\data\process\figs\\"+str(inte)+"\\"+d+".tif",dpi=300)
    
    
