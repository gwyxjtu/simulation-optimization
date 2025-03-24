'''
Author: guo_idpc
Date: 2023-02-24 15:03:18
LastEditors: guo-4060ti 867718012@qq.com
LastEditTime: 2025-03-24 10:27:09
FilePath: \总程序\model_HIES\model_load_day.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# from ast import main
import pandas as pd
import csv
from guo_method.guo_decorator import exception_handler
days=12

def get_all_scenario(data,day):
    # 按天切割全部场景
    return [data[i:i+int(len(data)/day)] for i in range(0, len(data), int(len(data)/day))]

@exception_handler
def get_data():

    book_water = pd.read_excel('./data/yulin_water_load.xlsx')
    g_demand = list(book_water['供暖热负荷(kW)'].fillna(0))
    water_load = list(book_water['生活热水负荷kW'].fillna(0))
    ele_load = list(book_water['电负荷kW'].fillna(0))

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
    

    g_scenario = get_all_scenario(g_demand,365)
    water_scenario = get_all_scenario(water_load,365)
    ele_scenario = get_all_scenario(ele_load,365)

    pv_5min_88,pv_5min_76,pv_5min_113 = get_all_scenario(pv_5min_88,365),get_all_scenario(pv_5min_76,365),get_all_scenario(pv_5min_113,365)
    pv_scenario_3 = [pv_5min_88,pv_5min_76,pv_5min_113]
    return g_scenario,water_scenario,ele_scenario,pv_scenario_3

def get_scenario_data(g_scenario,water_scenario,ele_scenario,pv_scenario_3):
    # 生成全年12个月的代表日索引（示例数据，需按实际数据结构调整）
    # month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
    # interval = 12 // days
    # scenario_days = [sum(month_days[:i*interval]) for i in range(days)]
    scenario_days = list(range(10,365,365//days))
    # scenario_days = [33,46,73,106,134,165,195,226,257,288,318,348]
    g_demand = [g_scenario[d] for d in scenario_days]
    # g_demand = [[2800 if g_demand[d][i] > 2800 else g_demand[d][i] for i in range(len(g_demand[d]))] for d in range(days)]
    water_load = [water_scenario[d] for d in scenario_days]
    ele_load = [ele_scenario[d] for d in scenario_days]
    pv_3 = [[pv_scenario_3[i][d] for d in scenario_days] for i in range(3)]
    return g_demand,water_load,ele_load,pv_3

def get_load():
    g_scenario,water_scenario,ele_scenario,pv_scenario_3 = get_data()
    g_demand,water_load,ele_load,pv_3 = get_scenario_data(g_scenario,water_scenario,ele_scenario,pv_scenario_3)
    
    return g_demand, ele_load, water_load, pv_3


if __name__ == "__main__":
    get_load()