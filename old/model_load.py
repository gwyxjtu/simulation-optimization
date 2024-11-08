'''
Author: guo_idpc
Date: 2023-02-24 15:03:18
LastEditors: guo_win 867718012@qq.com
LastEditTime: 2023-06-09 12:23:55
FilePath: /bilinear/main_model/model_load.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import pandas as pd
import csv
cer = 0.1
days=12



def crf(year):
    i = 0.08
    crf=((1+i)**year)*i/((1+i)**year-1);
    return crf
def get_load():


    ele_load = []
    g_demand = []
    q_demand = []
    r_solar = []


    #------------
    cost_fc = 15504
    cost_el = 9627.3
    cost_hst = 3600
    cost_eb = 434.21
    cost_water_hot = 1
    cost_pv = 900
    cost_pump = 730
    crf_fc = crf(10)
    crf_el = crf(7)
    crf_hst = crf(20)
    crf_water = crf(20)
    crf_pv = crf(20)
    crf_pump = crf(20)
    crf_eb = crf(15)

    #--------------
    lambda_ele_in = [0.3748,0.3748,0.3748,0.3748,0.3748,0.3748,0.3748,0.8745,0.8745,0.8745,1.4002,1.4002,1.4002,1.4002,
                    1.4002,0.8745,0.8745,0.8745,1.4002,1.4002,1.4002,0.8745,0.8745,0.3748]
    #lambda_ele_in = [lambda_ele_in[i]*1.5 for i in range(len(lambda_ele_in))]
    lambda_ele_out = 0
    #lambda_ele_in = lambda_ele_in*30

    ele_load = []
    g_demand = []
    q_demand = []
    r_solar = []

    book_cold = pd.read_excel('./data/cold.xlsx')
    book_heat = pd.read_excel('./data/heat.xlsx')
    for l in range(8760):
        q_demand.append(book_cold.iloc[l,1])
        ele_load.append(4/3*book_cold.iloc[l,1])

    # 热负荷从文章里面出
    # for l in range(2904):
    #     g_demand.append(3*book_heat.iloc[l,3])
    # g_demand = g_demand[:1128]+[0 for _ in range(8760-2904)]+g_demand[1128:]
    # 热负荷从榆林里面出，包括生活热水
    book_water = pd.read_excel('./data/yulin_water_load.xlsx')
    g_demand = list(book_water['供暖热负荷(kW)'].fillna(0))
    water_load = list(book_water['生活热水负荷kW'].fillna(0))

    # r_solar =  [0 for _ in range(8760)]
    with open("data/"+"solar.csv") as renewcsv:
        renewcsv.readline()
        renewcsv.readline()
        renewcsv.readline()
        renew = csv.DictReader(renewcsv)
        
        for row in renew:

            r_solar.append(float(row['electricity']))
    r_solar = r_solar[-8:]+r_solar[:-8]
    r_solar = r_solar[:8760]
    pv_5min_21 = pd.read_csv('data/pv_5min/Actual_40.85_-73.85_2006_DPV_21MW_5_Min.csv')['Power(MW)'].to_list()
    pv_5min_21 = [pv_5min_21[i]/21 for i in range(len(pv_5min_21))]
    pv_5min_76 = pd.read_csv('data/pv_5min/Actual_41.65_-74.25_2006_UPV_76MW_5_Min.csv')['Power(MW)'].to_list()
    pv_5min_76 = [pv_5min_76[i]/76 for i in range(len(pv_5min_76))]
    pv_5min_126 = pd.read_csv('data/pv_5min/Actual_41.25_-74.25_2006_UPV_126MW_5_Min.csv')['Power(MW)'].to_list()
    pv_5min_126 = [pv_5min_126[i]/126 for i in range(len(pv_5min_126))]
    pv_5min_88 = pd.read_csv('data/pv_5min/Actual_42.55_-74.15_2006_UPV_88MW_5_Min.csv')['Power(MW)'].to_list()
    pv_5min_88 = [pv_5min_88[i]/88 for i in range(len(pv_5min_88))]
    pv_5min_113 = pd.read_csv('data/pv_5min/Actual_42.85_-74.05_2006_UPV_113MW_5_Min.csv')['Power(MW)'].to_list()
    pv_5min_113 = [pv_5min_113[i]/113 for i in range(len(pv_5min_113))]

    m_date = [31,28,31,30,31,30,31,31,30,31,30,31]
    m_date = [sum(m_date[:i])*24 for i in range(12)]
    m_date.append(8760)
    # g_demand = [0 for i in range(8760)]
    # for h in [1,2,3,11,12]:
    #     g_demand[m_date[h-1]:m_date[h]] = [1 for _ in range(m_date[h]-m_date[h-1])]
    r=r_solar
    if days == 1:
        q_demand = q_demand[384+24:384+48]
        g_demand = g_demand[384+24:384+48]
        r_solar =   r_solar[384+24:384+48]
        ele_load = ele_load[384+24:384+48]
        #print(q_demand)
    elif days == 4:
        q_demand = q_demand[384+24:384+48]+q_demand[2496+48:2496+72]+q_demand[4680+48:4680+72]+q_demand[6888:6888+24]
        g_demand = g_demand[384+24:384+48]+g_demand[2496+48:2496+72]+g_demand[4680+48:4680+72]+g_demand[6888:6888+24]
        r_solar =   r_solar[384+24:384+48]+ r_solar[2496+48:2496+72]+ r_solar[4680+48:4680+72]+ r_solar[6888:6888+24]
        ele_load = ele_load[384+24:384+48]+ele_load[2496+48:2496+72]+ele_load[4680+48:4680+72]+ele_load[6888:6888+24]
        water_load = water_load[384+24:384+48]+water_load[2496+48:2496+72]+water_load[4680+48:4680+72]+water_load[6888:6888+24]
    elif days == 7:
        # 供热季的一天？
        q_demand = q_demand[24:8*24]
        g_demand = g_demand[24:8*24]
        r_solar =   r_solar[24:8*24]
        ele_load = ele_load[24:8*24]
    elif days == 12:
        g_demand = g_demand[384+24:384+48]+g_demand[1080+24:1080+48]+g_demand[1752:1752+24]+g_demand[2496+48:2496+72]+g_demand[3216:3216+24]+g_demand[3960:3960+24]+g_demand[4680+48:4680+72]+g_demand[5424:5424+24]+g_demand[6168:6168+24]+g_demand[6888+24:6888+48]+g_demand[7632:7632+24]+g_demand[8352:8352+24]
        q_demand = q_demand[384+24:384+48]+q_demand[1080+24:1080+48]+q_demand[1752:1752+24]+q_demand[2496+48:2496+72]+q_demand[3216:3216+24]+q_demand[3960:3960+24]+q_demand[4680+48:4680+72]+q_demand[5424:5424+24]+q_demand[6168:6168+24]+q_demand[6888+24:6888+48]+q_demand[7632:7632+24]+q_demand[8352:8352+24]
        r_solar =   r_solar[384+24:384+48]+ r_solar[1080+24:1080+48]+ r_solar[1752:1752+24]+ r_solar[2496+48:2496+72]+ r_solar[3216:3216+24]+ r_solar[3960:3960+24]+ r_solar[4680+48:4680+72]+ r_solar[5424:5424+24]+ r_solar[6168:6168+24]+r_solar[6888+24:6888+48]+r_solar[7632:7632+24]+r_solar[8352:8352+24]
        ele_load = ele_load[384+24:384+48]+ele_load[1080+24:1080+48]+ele_load[1752:1752+24]+ele_load[2496+48:2496+72]+ele_load[3216:3216+24]+ele_load[3960:3960+24]+ele_load[4680+48:4680+72]+ele_load[5424:5424+24]+ele_load[6168:6168+24]+ele_load[6888+24:6888+48]+ele_load[7632:7632+24]+ele_load[8352:8352+24]
        water_load = water_load[384+24:384+48]+water_load[1080+24:1080+48]+water_load[1752:1752+24]+water_load[2496+48:2496+72]+water_load[3216:3216+24]+water_load[3960:3960+24]+water_load[4680+48:4680+72]+water_load[5424:5424+24]+water_load[6168:6168+24]+water_load[6888+24:6888+48]+water_load[7632:7632+24]+water_load[8352:8352+24]
        pv_5min_21 = pv_5min_21[(384)*12:(384+24)*12]+pv_5min_21[(1080+24)*12:(1080+48)*12]+pv_5min_21[1752*12:(1752+24)*12]+pv_5min_21[(2496+48)*12:(2496+72)*12]+pv_5min_21[(3216+48)*12:(3216+72)*12]+pv_5min_21[3960*12:(3960+24)*12]+pv_5min_21[(4680+48)*12:(4680+72)*12]+pv_5min_21[5424*12:(5424+24)*12]+pv_5min_21[(6168+24)*12:(6168+48)*12]+pv_5min_21[(6888+24)*12:(6888+48)*12]+pv_5min_21[(7632+48)*12:(7632+72)*12]+pv_5min_21[(8352+24)*12:(8352+48)*12]
        pv_5min_76 = pv_5min_76[(384)*12:(384+24)*12]+pv_5min_76[(1080+24)*12:(1080+48)*12]+pv_5min_76[1752*12:(1752+24)*12]+pv_5min_76[(2496+48)*12:(2496+72)*12]+pv_5min_76[(3216+48)*12:(3216+72)*12]+pv_5min_76[3960*12:(3960+24)*12]+pv_5min_76[(4680+48)*12:(4680+72)*12]+pv_5min_76[5424*12:(5424+24)*12]+pv_5min_76[(6168+24)*12:(6168+48)*12]+pv_5min_76[(6888+24)*12:(6888+48)*12]+pv_5min_76[(7632+48)*12:(7632+72)*12]+pv_5min_76[(8352+24)*12:(8352+48)*12]
        pv_5min_126 = pv_5min_126[(384)*12:(384+24)*12]+pv_5min_126[(1080+24)*12:(1080+48)*12]+pv_5min_126[1752*12:(1752+24)*12]+pv_5min_126[(2496+48)*12:(2496+72)*12]+pv_5min_126[(3216+48)*12:(3216+72)*12]+pv_5min_126[3960*12:(3960+24)*12]+pv_5min_126[(4680+48)*12:(4680+72)*12]+pv_5min_126[5424*12:(5424+24)*12]+pv_5min_126[(6168+24)*12:(6168+48)*12]+pv_5min_126[(6888+24)*12:(6888+48)*12]+pv_5min_126[(7632)*12:(7632+24)*12]+pv_5min_126[(8352+24)*12:(8352+48)*12]
        pv_5min_113 = pv_5min_113[(384)*12:(384+24)*12]+pv_5min_113[(1080+24)*12:(1080+48)*12]+pv_5min_113[1752*12:(1752+24)*12]+pv_5min_113[(2496+48)*12:(2496+72)*12]+pv_5min_113[(3216+48)*12:(3216+72)*12]+pv_5min_113[3960*12:(3960+24)*12]+pv_5min_113[(4680+48)*12:(4680+72)*12]+pv_5min_113[5424*12:(5424+24)*12]+pv_5min_113[(6168+24)*12:(6168+48)*12]+pv_5min_113[(6888+24)*12:(6888+48)*12]+pv_5min_113[(7632)*12:(7632+24)*12]+pv_5min_113[(8352+24)*12:(8352+48)*12]
        pv_5min_88 = pv_5min_88[(384)*12:(384+24)*12]+pv_5min_88[(1080+24)*12:(1080+48)*12]+pv_5min_88[1752*12:(1752+24)*12]+pv_5min_88[(2496+48)*12:(2496+72)*12]+pv_5min_88[(3216+48)*12:(3216+72)*12]+pv_5min_88[3960*12:(3960+24)*12]+pv_5min_88[(4680+48)*12:(4680+72)*12]+pv_5min_88[5424*12:(5424+24)*12]+pv_5min_88[(6168+24)*12:(6168+48)*12]+pv_5min_88[(6888+24)*12:(6888+48)*12]+pv_5min_88[(7632)*12:(7632+24)*12]+pv_5min_88[(8352+24)*12:(8352+48)*12]
    elif days == 24:
        g_demand = g_demand[384+24:384+48]+g_demand[1080+24:1080+48]+g_demand[1752:1752+24]+g_demand[2496+48:2496+72]+g_demand[3216:3216+24]+g_demand[3960:3960+24]+g_demand[4680+48:4680+72]+g_demand[5424:5424+24]+g_demand[6168:6168+24]+g_demand[6888+24:6888+48]+g_demand[7632:7632+24]+g_demand[8352:8352+24]
        q_demand = q_demand[384+24:384+48]+q_demand[1080+24:1080+48]+q_demand[1752:1752+24]+q_demand[2496+48:2496+72]+q_demand[3216:3216+24]+q_demand[3960:3960+24]+q_demand[4680+48:4680+72]+q_demand[5424:5424+24]+q_demand[6168:6168+24]+q_demand[6888+24:6888+48]+q_demand[7632:7632+24]+q_demand[8352:8352+24]
        r_solar =   r_solar[384+24:384+48]+ r_solar[1080+24:1080+48]+ r_solar[1752:1752+24]+ r_solar[2496+48:2496+72]+ r_solar[3216:3216+24]+ r_solar[3960:3960+24]+ r_solar[4680+48:4680+72]+ r_solar[5424:5424+24]+ r_solar[6168:6168+24]+r_solar[6888+24:6888+48]+r_solar[7632:7632+24]+r_solar[8352:8352+24]
        ele_load = ele_load[384+24:384+48]+ele_load[1080+24:1080+48]+ele_load[1752:1752+24]+ele_load[2496+48:2496+72]+ele_load[3216:3216+24]+ele_load[3960:3960+24]+ele_load[4680+48:4680+72]+ele_load[5424:5424+24]+ele_load[6168:6168+24]+ele_load[6888+24:6888+48]+ele_load[7632:7632+24]+ele_load[8352:8352+24]
        water_load = water_load[384+24:384+48]+water_load[1080+24:1080+48]+water_load[1752:1752+24]+water_load[2496+48:2496+72]+water_load[3216:3216+24]+water_load[3960:3960+24]+water_load[4680+48:4680+72]+water_load[5424:5424+24]+water_load[6168:6168+24]+water_load[6888+24:6888+48]+water_load[7632:7632+24]+water_load[8352:8352+24]


        g_demand = g_demand+g_demand
        q_demand = q_demand+q_demand
        r_solar = r_solar+r_solar
        ele_load = ele_load+ele_load
        water_load = water_load+water_load
    elif days == 36:
        g_demand = g_demand[384+24:384+48]+g_demand[1080+24:1080+48]+g_demand[1752:1752+24]+g_demand[2496+48:2496+72]+g_demand[3216:3216+24]+g_demand[3960:3960+24]+g_demand[4680+48:4680+72]+g_demand[5424:5424+24]+g_demand[6168:6168+24]+g_demand[6888+24:6888+48]+g_demand[7632:7632+24]+g_demand[8352:8352+24]
        q_demand = q_demand[384+24:384+48]+q_demand[1080+24:1080+48]+q_demand[1752:1752+24]+q_demand[2496+48:2496+72]+q_demand[3216:3216+24]+q_demand[3960:3960+24]+q_demand[4680+48:4680+72]+q_demand[5424:5424+24]+q_demand[6168:6168+24]+q_demand[6888+24:6888+48]+q_demand[7632:7632+24]+q_demand[8352:8352+24]
        r_solar =   r_solar[384+24:384+48]+ r_solar[1080+24:1080+48]+ r_solar[1752:1752+24]+ r_solar[2496+48:2496+72]+ r_solar[3216:3216+24]+ r_solar[3960:3960+24]+ r_solar[4680+48:4680+72]+ r_solar[5424:5424+24]+ r_solar[6168:6168+24]+r_solar[6888+24:6888+48]+r_solar[7632:7632+24]+r_solar[8352:8352+24]
        ele_load = ele_load[384+24:384+48]+ele_load[1080+24:1080+48]+ele_load[1752:1752+24]+ele_load[2496+48:2496+72]+ele_load[3216:3216+24]+ele_load[3960:3960+24]+ele_load[4680+48:4680+72]+ele_load[5424:5424+24]+ele_load[6168:6168+24]+ele_load[6888+24:6888+48]+ele_load[7632:7632+24]+ele_load[8352:8352+24]
        water_load = water_load[384+24:384+48]+water_load[1080+24:1080+48]+water_load[1752:1752+24]+water_load[2496+48:2496+72]+water_load[3216:3216+24]+water_load[3960:3960+24]+water_load[4680+48:4680+72]+water_load[5424:5424+24]+water_load[6168:6168+24]+water_load[6888+24:6888+48]+water_load[7632:7632+24]+water_load[8352:8352+24]

        g_demand = g_demand+g_demand+g_demand
        q_demand = q_demand+q_demand+q_demand
        r_solar = r_solar+r_solar+r_solar
        ele_load = ele_load+ele_load+ele_load
        water_load = water_load+water_load+water_load
    elif days == 365:
        1
    # import matplotlib.pyplot as plt
    # # print(g_demand)
    # # print(q_demand)
    # x = [i for i in range(0,24*days)]
    # plt.plot(x,g_demand)
    # plt.plot(x,q_demand)
    # plt.plot(x,water_load)
    # plt.plot(x,ele_load)
    # plt.show()
    # plt.savefig('img/g_de.png')
    # exit(0)

    #g_de = g_de_w*days
    #p_load = p_load_winter*days
    lambda_ele_in = lambda_ele_in*days
    #r = r*days
    # m_de = [g_de[i]/c_kWh/delta_T for i in range(len(g_de))]

    # g_demand = [g_demand[i] for i in range(len(g_demand))]
    # water_load = [water_load[i] for i in range(len(water_load))]
    # 光伏需要总量归一化，描述波动
    # sum_r = sum(pv_5min_21)
    # sum_76 = sum(pv_5min_76)
    # sum_126 = sum(pv_5min_126)

    # pv_5min_76 = [pv_5min_76[i]/sum_76*sum_r for i in range(len(pv_5min_76))]
    # pv_5min_126 = [pv_5min_126[i]/sum_126*sum_r for i in range(len(pv_5min_126))]
    # pv_5min_126 = [pv_5min_126[i]*1.07 for i in range(len(pv_5min_126))]
    pv_5min_113 = [pv_5min_113[i]*1.1 for i in range(len(pv_5min_126))]
    # pv_5min_21 = [pv_5min_21[i]*1.1 for i in range(len(pv_5min_21))]
    pv_5min_88 = [pv_5min_88[i]*0.95 for i in range(len(pv_5min_88))]
    pv_5min_76 = [pv_5min_88[i]*1.05 for i in range(len(pv_5min_88))]
    period = len(g_demand)
    print(sum(pv_5min_88),sum(pv_5min_76),sum(pv_5min_113))
    import numpy as np
    print(np.std(pv_5min_88),np.std(pv_5min_76),np.std(pv_5min_113))
    pv_5min_113[151:153]=[0.17,0.19]
    return lambda_ele_in, g_demand, q_demand, r_solar, [pv_5min_88,pv_5min_76,pv_5min_113], ele_load, water_load, period