'''
Author: guo_MateBookPro 867718012@qq.com
Date: 2023-07-19 16:05:26
LastEditors: guo-4060ti 867718012@qq.com
LastEditTime: 2025-03-25 11:08:59
FilePath: \总程序\main_opt.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.
Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import coptpy as cp
from coptpy import COPT
import pandas as pd
from model_HIES.scenario_calss import Scenario_all_day,Scenario
from model_HIES.model_load_day import days,get_load
from model_HIES.model_class import MultiTime_model
from guo_method.mymail import send
from guo_method.guo_decorator import exception_handler,timer
import pprint
import multiprocessing as mp
import xlwt

wb = xlwt.Workbook()
capacity=wb.add_sheet('容量记录')

capacity.write(0,0,'P_fc')
capacity.write(0,1,'P_el')
capacity.write(0,2,'P_pv')
capacity.write(0,3,'P_hp')
capacity.write(0,4,'P_bs')
capacity.write(0,5,'H_hs')
capacity.write(0,6,'P_eb')
capacity.write(0,7,'G_ht')



receivers = ['guoguoloveu@icloud.com']


def main_alg(pv_number:int, pv_lb:int, ramp_rate:float) -> None:

    iter=0

    """_summary_

    Args:
        pv_number (int): _description_
        ramp_rate (float): _description_
    """
    all_scenario = Scenario_all_day(days,pv_number,*get_load())
    MultiTime_m = MultiTime_model(pv_lb, ramp_rate) 
    MP = MultiTime_m.formulate_MP(all_scenario.main_scenario)
    MultiTime_m.solve_MP(wb,capacity)
    MultiTime_m.get_MP_result()
    fail_case = days
    operation_revise = {'fail':False}

    error = pd.DataFrame(columns=[f'Scenario{i+1}' for i in range(all_scenario.days)])
    while fail_case > 0:
    #while iter<=200:
        fail_case = 0
        # for sub_s in all_scenario.sub_scenario:
        error_s = pd.DataFrame(columns=[f'Scenario{i+1}' for i in range(all_scenario.days)])
        for i in range(all_scenario.days):
            operation_revise = MultiTime_m.simulation_heat(MultiTime_m.operation_res[i],MultiTime_m.operation_res_5min[i],all_scenario.sub_scenario[i],operation_revise,i+1,iter)
            if operation_revise['fail'] == False:
                continue
            
            error_sum = MultiTime_m.add_cut(MultiTime_m.operation_res[i],all_scenario.sub_scenario[i],operation_revise,i)
            error_s.loc[iter, f'Scenario{i+1}'] = error_sum
            
            fail_case += 1

        iter=iter+1
        error = pd.concat([error, error_s])
        
        error.to_csv("res/error.csv")
        MultiTime_m.solve_MP(wb,capacity,iter)
        MultiTime_m.get_MP_result(iter)
    if iter>days*30:
        send('计算完毕',receivers,"ok",["res_data/device_cap.csv"])
        

# def main_alg_multiprocess(pv_number:int,ramp_rate:float) -> None:
#     """_summary_

#     Args:
#         pv_number (int): _description_
#         ramp_rate (float): _description_
#     """
#     all_scenario = Scenario_all_day(days,pv_number,*get_load())
#     MultiTime_m = MultiTime_model(ramp_rate)
#     MP = MultiTime_m.formulate_MP(all_scenario.main_scenario)
#     MultiTime_m.solve_MP()
    
#     fail_case = days

#     while fail_case > 0:
#         fail_case = 0
#         process_list = []
#         for sub_s in all_scenario.sub_scenario:
#             p = mp.Process(target=MultiTime_m.simulation_heat,args=(sub_s,))
#             p.start()
#             p.join()
#             process_list.append(p)
 
     
       
        

if __name__ == '__main__':



    main_alg(1,1100,1/6)


    # pd.DataFrame(dict([(k, pd.Series(v)) for k, v in res.items()])).to_csv("res_data/test.csv")
    # pd.DataFrame(device_cap,index=[0]).to_csv("res_data/device_cap.csv")
    # send('计算完毕',receivers,"ok",["res_data/device_cap.csv"])