'''
Author: guo_win 867718012@qq.com
Date: 2023-06-01 19:09:09
LastEditors: working-guo 867718012@qq.com
LastEditTime: 2023-07-19 11:13:16
FilePath: \copt_multi-time\old\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pprint
import numpy as np
import pandas as pd
# from pyrsistent import v
from guo_method.guo_decorator import exception_handler,timer
from old.model_load import *
import coptpy as cp
from coptpy import COPT
def re(year, life):
    l = [0 for _ in range(life)]
    for i in range(life):
        if i in [year, 2 * year, 3 * year, 4 * year, 5 * year]:
            l[i] = 1
    return l
# def get_para():
#     lambda_ele_in = [0.3748,0.3748,0.3748,0.3748,0.3748,0.3748,0.3748,0.8745,0.8745,0.8745,1.4002,1.4002,1.4002,1.4002,
#                     1.4002,0.8745,0.8745,0.8745,1.4002,1.4002,1.4002,0.8745,0.8745,0.3748]
#     lambda_ele_in = lambda_ele_in*days
#     return lambda_ele_in
    
    
def opt(ramp_up,pv_osi):
    """
    discription: 优化主函数
    param {*}
    return {*}
    """
    M = 1000000
    d = 0.08  #利率
    mu_OM = 0.02 #每年运维成本系数
    
    pv_lb = 6000
    # pv_osi = 0 # 0 1 2 波动程度
    cer = 0.6
    # 储能配比参数
    delta_ru = 0.2
    hour = 4
    # fc爬坡参数
    # ramp_up = 1/1 # 15min 完全爬坡，关停的话可以直接关停
    pur_cap = 0
    # 投资参数
    cost_fc = 8000
    cost_el = 2240
    cost_hst = 3600
    cost_eb = 434.21
    cost_pv = 5000
    cost_bs = 2300
    cost_hp = 3000
    cost_ghp = 8000
    cost_ec = 500
    cost_ht = 40 # 储热
    system_life = 20

    pi_EL = re(10, system_life)
    pi_FC = re(10, system_life)
    pi_BS = re(4, system_life)


    # 系数
    k_fc = 16
    k_el = 45
    k_hp = 3.4
    k_ghp_g = 4
    k_ghp_q = 5
    eta_fc = 1.2
    k_ec = 3
    eta_bs = 0.99
    eta_ht = 0.9
    soc_min = 0.15
    soc_max = 0.9
    bs_hour = 2
    cooling_rate = 1# p=rate*c
    c = 4200 # J/kg*C
    c_kWh = 4200/3.6/1000000
    lambda_h = 25

    # 读取数据
    # lambda_ele_in, g_demand, q_demand, r_solar, pv_5min, ele_load, water_load, period 
    
    g_demand, ele_load, water_load, pv_3 = get_load()
    period = len(g_demand)*24
    period_ele = period*12
    pv_5min = pv_3[pv_osi]
    ele_peak = 1500
    ele_idle = 1500*0.6
    it_rt = [0.5,0.4,0.2,0,0.1,0.3,0.4,0.5,0.7,0.8,1,0.8,0.8,0.7,0.7,0.6,0.4,0.3,0.1,0.2,0.3,0.5,0.4,0.3]*days
    it_rt = [it_rt[i]/2 for i in range(len(it_rt))]
    it_dt = it_rt
    # it_dt = [0 for _ in range(period)]
    it_load_max = 1
    c_dt_max = 2
    

    # 优化
    env = cp.Envr()
    model = env.createModel("idc")
    
    # 优化变量
    C_IN = model.addVar(vtype=COPT.CONTINUOUS, name = "capex")
    opex = model.addVar(vtype=COPT.CONTINUOUS, name = "opex")
    C_OM = model.addVar(vtype=COPT.CONTINUOUS, name = "C_OM")
    C_RE = model.addVar(vtype=COPT.CONTINUOUS, name = "C_RE")
    P_fc = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_fc")
    P_el = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_el")
    P_pv = model.addVar( lb = pv_lb, vtype=COPT.CONTINUOUS, name = "P_pv")
    P_hp = model.addVar( lb=0,ub=500, vtype=COPT.CONTINUOUS, name = "P_hp")
    P_ghp = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "p_ghp")
    P_bs = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_bt")
    H_sto = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "H_sto")
    P_ec = model.addVar( ub=0, vtype=COPT.CONTINUOUS, name = "P_ec")
    G_ht = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "G_ht")

    z_el = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_el")
    z_fc = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_fc")

    p_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_fc")
    g_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "g_fc")
    h_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "h_fc")
    g_fc_hour = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_fc_hour")

    h_el = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "h_el")
    p_el = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_el")
    h_sto = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "h_sto")
    h_pur = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "h_pur")

    p_ghp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_ghp")
    g_ghp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ghp")
    q_ghp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "q_ghp")
    z_ghp_g = model.addVars(period, vtype = COPT.BINARY, nameprefix = "z_ghp_g")
    z_ghp_q = model.addVars(period, vtype = COPT.BINARY, nameprefix = "z_ghp_q")


    p_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_hp")
    q_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "q_hp")
    g_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_hp")

    p_pur = model.addVars(period_ele,ub=pur_cap, vtype = COPT.CONTINUOUS, nameprefix = "p_pur")
    p_sol = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_sol")

    c_dt = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "c_dt")
    it_dt_n = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "it_dt_n")
    it_load = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "it_load")
    
    q_demand = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "q_demand")
    ele_load = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "ele_load")
    p_pv = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_pv")

    #soc = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "soc") # 蓄电池
    p_bs_di = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bt_di") # 蓄电池放电
    p_bs_ch = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bt_ch") # 蓄电池充电
    p_bs = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bt") # 蓄电池储电量

    p_ec = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_ec") # 冷水机组耗电
    q_ec = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "q_ec") # 冷水机组制冷
    # 储热
    g_ht = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ht") # 储热
    g_ht_ch = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ht_ch") # 储热充电
    g_ht_di = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_gt_di") # 储热放电

    # 储能约束,储氢，蓄电池,地热平衡
    model.addConstrs(g_ht[i+1] - g_ht[i] == g_ht_ch[i]*eta_ht - g_ht_di[i]/eta_ht for i in range(period - 1))
    model.addConstrs(h_sto[i+1] - h_sto[i] == h_pur[i] + h_el[i] - h_fc[i] for i in range(period - 1))
    model.addConstrs(p_bs[i+1] - p_bs[i] == p_bs_ch[i]*eta_bs - p_bs_di[i]/eta_bs for i in range(period_ele - 1))
    
    model.addConstr(g_ht[0] - g_ht[period-1] == g_ht_ch[period - 1]*eta_ht - g_ht_di[period - 1]/eta_ht)
    model.addConstr(h_sto[0] - h_sto[period-1] == h_pur[period-1] + h_el[period-1] - h_fc[period-1])
    model.addConstr(p_bs[0] - p_bs[period_ele-1] == p_bs_ch[period_ele-1]*eta_bs - p_bs_di[period_ele-1]/eta_bs)
    
    model.addConstrs(p_bs[i] <= soc_max * P_bs for i in range(period_ele))
    model.addConstrs(p_bs[i] >= soc_min * P_bs for i in range(period_ele))
    model.addConstrs(p_bs_ch[i] <= P_bs/bs_hour for i in range(period_ele))
    model.addConstrs(p_bs_di[i] <= P_bs/bs_hour for i in range(period_ele))
    model.addConstrs(h_sto[i] <= H_sto for i in range(period))

    model.addConstrs(g_ht[i*24+24] == g_ht[i*24] for i in range(period//24-1))

    model.addConstr(g_ghp.sum()<=q_ghp.sum()+p_ghp.sum())

    # 设备约束
    # ghp
    # model.addConstrs(g_demand[i]<=M*z_ghp_g[i] for i in range(period))
    model.addConstrs(M*z_ghp_g[i]>=g_ghp[i] for i in range(period))
    model.addConstrs(M*z_ghp_q[i]>=q_ghp[i] for i in range(period))
    model.addConstrs(z_ghp_g[i]+z_ghp_q[i]<=1 for i in range(period))
    model.addConstrs(p_ghp[i] == g_ghp[i]/k_ghp_g + q_ghp[i]/k_ghp_q for i in range(period))
    # waste heat pump
    model.addConstrs(k_hp*p_hp[i] == g_hp[i] for i in range(period))
    model.addConstrs(q_hp[i] == g_hp[i]*(1-1/k_hp) for i in range(period))
    # fuel cell
    model.addConstrs(g_fc[i] == eta_fc*k_fc*h_fc[i] for i in range(period_ele))
    model.addConstrs(p_fc[i] == k_fc*h_fc[i] for i in range(period_ele))
    model.addConstrs(g_fc[i+1] - g_fc[i] <= ramp_up*P_fc*eta_fc for i in range(period_ele-1))
    model.addConstrs(g_fc_hour[i] == cp.quicksum(g_fc[i*12+j] for j in range(12)) for i in range(period))
    # el
    model.addConstrs(p_el[i] == h_el[i]*k_el for i in range(period))
    # model.addConstrs(M*z_fc[i] >= p_fc[i] for i in range(period))
    # model.addConstrs(M*z_el[i] >= p_el[i] for i in range(period))
    # model.addConstrs(z_fc[i] + z_el[i] <= 1 for i in range(period))
    model.addConstrs(h_el[i] <= H_sto for i in range(period))
    # ec
    model.addConstrs(q_ec[i] == p_ec[i]*k_ec for i in range(period))
    # pv
    model.addConstrs(p_pv[i]==pv_5min[i]*P_pv for i in range(period_ele))
    # # 储能配比
    # model.addConstr(P_bs / bs_hour + P_el >= P_pv * delta_ru)
    # model.addConstr(P_bs / bs_hour + P_fc >= P_pv * delta_ru)
    # model.addConstr(P_bs + H_sto / k_fc >= P_pv * delta_ru * hour)
    # # heat supply
    model.addConstrs(g_fc_hour[i]+g_hp[i]+g_ghp[i]+g_ht_di[i] == g_demand[i]+g_ht_ch[i] + water_load[i] for i in range(period))
    # cooling supply
    model.addConstrs(q_hp[i] + q_ghp[i] + q_ec[i] == q_demand[i] for i in range(period))
    # power supply
    model.addConstrs(p_el[i//12]  + ele_load[i//12] + p_ghp[i//12] + p_hp[i//12] + p_ec[i//12] + p_bs_ch[i] == p_pur[i] + p_fc[i] + p_pv[i] + p_bs_di[i] for i in range(period_ele))
    

    # it_load
    model.addConstrs(c_dt[i+1] == c_dt[i] + it_dt[i] - it_dt_n[i+1] for i in range(period - 1))
    model.addConstrs(c_dt[i*24] == 0 for i in range(int(period/24)))
    model.addConstr(c_dt[period-1] == 0)

    model.addConstrs(q_demand[i] == cooling_rate*ele_load[i] for i in range(period))
    model.addConstrs(ele_load[i] == ele_idle + (ele_peak - ele_idle) * it_load[i] for i in range(period))
    model.addConstrs(it_load[i] == it_rt[i] + it_dt_n[i] for i in range(period))
    model.addConstrs(it_load[i] <= it_load_max for i in range(period))
    model.addConstrs(c_dt[i] <= c_dt_max for i in range(period))
    
    # 容量上限
    model.addConstrs(p_fc[i] <= P_fc for i in range(period_ele))
    model.addConstrs(p_hp[i] <= P_hp for i in range(period))
    model.addConstrs(p_ghp[i] <= P_ghp for i in range(period))
    model.addConstrs(p_el[i] <= P_el for i in range(period))
    model.addConstrs(p_ec[i] <= P_ec for i in range(period))
    model.addConstrs(g_ht[i] <= G_ht for i in range(period))
    # cer
    model.addConstr(p_pur.sum()<=(1-cer)*(ele_load.sum()+sum(g_demand)/0.9+q_demand.sum()/4))
    # 优化目标
    # model.addConstr(capex == crf_ec*cost_ec*P_ec + crf_bs*cost_bs*P_bs + crf_pv*cost_pv*P_pv + crf_fc*cost_fc*P_fc + crf_hp*cost_hp*P_hp + crf_ghp*cost_ghp*P_ghp + crf_hst*cost_hst*H_sto + crf_el*cost_el*P_el)
    model.addConstr(C_IN == cost_ht*G_ht + cost_ec*P_ec + cost_bs*P_bs + cost_pv*P_pv + cost_fc*P_fc + cost_hp*P_hp + cost_ghp*P_ghp + cost_hst*H_sto + cost_el*P_el)
    model.addConstr(C_OM == cp.quicksum(mu_OM * C_IN / pow((1 + d), (i + 1)) for i in range(system_life)))
    model.addConstr(C_RE == cp.quicksum(
        (pi_EL[i] * cost_el * P_el + pi_FC[i] * cost_fc * P_fc + pi_BS[i] * cost_bs * P_bs) / pow((1 + d), (i + 1)) for i in
        range(system_life)))
    model.addConstr(opex == (365/days)*(cp.quicksum([lambda_ele_in[i//12]*p_pur[i] for i in range(period_ele)])+ h_pur.sum()*lambda_h) )
    model.setObjective(crf(system_life) * (C_IN + C_OM + C_RE) + opex , sense=COPT.MINIMIZE)
    # 求解参数
    model.setParam(COPT.Param.RelGap, 0.05)
    # model.setParam(COPT.Param.Threads, 128)
    model.setParam(COPT.Param.MipTasks,256)
    model.solve()
    if model.status == 2:
        model.computeIIS() 
        model.writeIIS("example.iis")
        # if model.hasIIS:
        #     # Print variables and constraints in IIS
        #     cons = model.getConstrs()
        #     vars = model.getVars()

        #     print("\n======================== IIS result ========================")
        #     for con in cons:
        #         if con.iislb or con.iisub:
        #             print('  {0}: {1}'.format(con.name, "lower" if con.iislb else "upper"))

        #     print("")
        #     for var in vars:
        #         if var.iislb or var.iisub:
        #             print('  {0}: {1}'.format(var.name, "lower" if var.iislb else "upper"))

        #     # Write IIS to file
        #     print("")
        #     model.writeIIS('iis_ex1.iis')

    # if model.status == COPT.OPTIMAL:
    #     print('Objective value: {}'.format(model.objval))
    #     print('Solving Time: {} seconds'.format(model.SolvingTime))
        
    #     allvars = model.getVars()
        
    #     print('Variable solution:')
    #     for var in allvars:
    #         print('{0}: {1}'.format(var.name, var.x))
            
    #     print('Variable basis status:')
    #     for var in allvars:
    #         print('{0}: {1}'.format(var.name, var.basis))

    # 处理结果
    device_cap = {
        "PV": P_pv.x,
        "ramp_up": str(int(1/ramp_up)),
        "cer":1 - sum([p_pur[i].x for i in range(period_ele)])/(sum([ele_load[i].x +q_demand[i].x/4 for i in range(period)])+sum(g_demand)/0.9)/12,
        "cap":pur_cap,
        "PV_osci":pv_osi,
        "obj": model.objval,
        "cap_total": C_IN.x,
        "opex": opex.x,
        "p_pur_sum": p_pur.sum().getValue(),
        "h_pur_sum": h_pur.sum().getValue(),
        "G_ht": G_ht.x,
        "P_pv": P_pv.x,
        "P_ec": P_ec.x,
        "P_bs": P_bs.x,
        "P_fc": P_fc.x,
        "P_el": P_el.x,
        "P_hp": P_hp.x,
        "P_ghp": P_ghp.x,
        "H_sto": H_sto.x
    }
    res_8760 = {
        "ele_load": [ele_load[i].x for i in range(period)],
        "p_el": [p_el[i].x for i in range(period)],
        "p_hp": [p_hp[i].x for i in range(period)],
        "p_ghp": [p_ghp[i].x for i in range(period)],
        
        "p_ec": [p_ec[i].x for i in range(period)],

        
        "g_hp": [g_hp[i].x for i in range(period)],
        "g_ghp": [g_ghp[i].x for i in range(period)],
        "g_fc_hour": [g_fc_hour[i].x for i in range(period)],
        "g_demand": g_demand,
        "water_load": water_load,
        "total_g_load":[g_demand[i]+water_load[i] for i in range(period)],
        
        "q_hp": [q_hp[i].x for i in range(period)],
        "q_ghp": [q_ghp[i].x for i in range(period)],
        "q_ec": [q_ec[i].x for i in range(period)],
        "q_demand": [q_demand[i].x for i in range(period)],

        "c_dt": [c_dt[i].x for i in range(period)],
        "it_dt_n": [it_dt_n[i].x for i in range(period)],
        "it_load": [it_load[i].x for i in range(period)],

        # "z_fc": [z_fc[i].x for i in range(period)],
        # "z_el": [z_el[i].x for i in range(period)],
        
        
        "h_el": [h_el[i].x for i in range(period)],
        "h_sto": [h_sto[i].x for i in range(period)],
        "h_pur": [h_pur[i].x for i in range(period)],

    }
    
    res_5min = {
        
        "ele_load": [ele_load[i//12].x for i in range(period_ele)],
        "p_el": [p_el[i//12].x for i in range(period_ele)],
        "p_hp": [p_hp[i//12].x for i in range(period_ele)],
        "p_ghp": [p_ghp[i//12].x for i in range(period_ele)],
        "p_sol": [p_sol[i].x for i in range(period_ele)],
        "p_ec": [p_ec[i//12].x for i in range(period_ele)],

        "p_pv": [p_pv[i].x for i in range(period_ele)],
        "p_pur": [p_pur[i].x for i in range(period_ele)],
        "p_fc": [p_fc[i].x for i in range(period_ele)],
        "r_solar":pv_5min,

        "p_bs": [p_bs[i].x for i in range(period_ele)],
        "p_bs_ch": [p_bs_ch[i].x for i in range(period_ele)],
        "p_bs_di": [p_bs_di[i].x for i in range(period_ele)],

        "g_ht": [g_ht[i//12].x for i in range(period_ele)],
        "g_ht_ch": [g_ht_ch[i//12].x for i in range(period_ele)],
        "g_ht_di": [g_ht_di[i//12].x for i in range(period_ele)],
        "g_hp": [g_hp[i//12].x for i in range(period_ele)],
        "g_ghp": [g_ghp[i//12].x for i in range(period_ele)],
        "g_fc": [g_fc[i].x for i in range(period_ele)],
        "g_fc_hour": [g_fc_hour[i//12].x for i in range(period_ele)],
        "g_demand": [g_demand[i//12] for i in range(period_ele)],
        "water_load": [water_load[i//12] for i in range(period_ele)],
        "total_g_load":[g_demand[i//12]+water_load[i//12] for i in range(period_ele)],
        
        "q_hp": [q_hp[i//12].x for i in range(period_ele)],
        "q_ghp": [q_ghp[i//12].x for i in range(period_ele)],
        "q_ec": [q_ec[i//12].x for i in range(period_ele)],
        "q_demand": [q_demand[i//12].x for i in range(period_ele)],

        "c_dt": [c_dt[i//12].x for i in range(period_ele)],
        "it_dt_n": [it_dt_n[i//12].x for i in range(period_ele)],
        "it_load": [it_load[i//12].x for i in range(period_ele)],

        
        "h_fc": [h_fc[i].x for i in range(period_ele)],
        "h_el": [h_el[i//12].x for i in range(period_ele)],
        "h_sto": [h_sto[i//12].x for i in range(period_ele)],
        "h_pur": [h_pur[i//12].x for i in range(period_ele)],

    }
    pd.DataFrame(res_5min).to_csv("res_data/PV"+str(pv_lb)+"KW-ramp"+str(int(1/ramp_up))+"time-cap"+str(pur_cap)+"pvosi-"+str(pv_osi)+".csv")
    pd.DataFrame(device_cap,index=[0]).to_csv("res_data/PV"+str(pv_lb)+"KW-ramp"+str(int(1/ramp_up))+"time-cap"+str(pur_cap)+"pvosi-"+str(pv_osi)+"-devicecap.csv")
    pprint.pprint(device_cap)
    return device_cap,res_5min