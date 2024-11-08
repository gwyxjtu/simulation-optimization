'''
Author: guo-4060ti 867718012@qq.com
Date: 2024-04-19 16:42:23
LastEditors: guo_MateBookPro 867718012@qq.com
LastEditTime: 2024-04-20 16:10:28
FilePath: /copt_multi-time/heat_simulation/simulation_device.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.
Copyright (c) 2024 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
from hplib import hplib as hpl
from opem.Static.Amphlett import Static_Analysis
import pandas as pd
import numpy as np
def simulation_hp(t_out_simulation):
    """输入24小时的出水温度序列返回COP

    Args:
        t_out (_type_): _description_

    Returns:
        _type_: _description_
    """
    parameters = hpl.get_parameters('Generic', group_id=1, t_in=10, t_out=50, p_th = 10000)
    heatpump=hpl.HeatPump(parameters)
    results = heatpump.simulate(t_in_primary=np.array([7 for _ in range(len(t_out_simulation))]), t_in_secondary=np.array([i-5 for i in t_out_simulation]), t_amb=np.array([20 for _ in range(len(t_out_simulation))]), mode=1)
    return results['COP']

def simulation_fc(P_fc, p_fc):
    """不应该直接cut电热功率，应该对效率进行cut，返回效率和电热比

    Args:
        P_fc (_type_): _description_
        p_fc (_type_): _description_

    Returns:
        _type_: _description_
    """
    Test_Vector={
        "T": 343.15,
        "PH2": 1,
        "PO2": 1,
        "i-start": 10,
        "i-stop": 300,
        "i-step": 0.1,
        "A": 300.6,
        "l": 0.0178,
        "lambda": 23,
        "N": 48,
        "R": 0,
        "JMax": 1.5,
        "Name": "Amphlett_Test"}
    data=Static_Analysis(InputMethod=Test_Vector,TestMode=True,PrintMode=True,ReportMode=False)
    fc_df = pd.DataFrame(data)
    eff, Electrothermal = np.array([]), np.array([])
    for i in range(len(p_fc)):
        index = (fc_df['P']-p_fc[i]/P_fc*10*1000).abs().idxmin()
        eff = np.append(eff, fc_df['EFF'][index])
        Electrothermal = np.append(Electrothermal, fc_df['Ph'][index]/fc_df['P'][index])
    # index = (fc_df['P']-p_fc/P_fc*10*1000).abs().idxmin()
    # g_fc = fc_df['Ph'][index]*p_fc/P_fc*10*1000
    # eff = fc_df['EFF'][index]
    return eff, Electrothermal
if __name__ == '__main__':
    # print(simulation_hp([50 for _ in range(24)]))
    print(simulation_fc(600, [500 for _ in range(24)]))
