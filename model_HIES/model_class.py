'''
Author: working-guo 867718012@qq.com
Date: 2023-07-18 16:03:36
LastEditors: guo_MateBookPro 867718012@qq.com
LastEditTime: 2024-11-08 15:46:02
FilePath: /总程序/model_HIES/model_class.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.
Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import pprint
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
from guo_method.guo_decorator import exception_handler,timer
from heat_simulation.simulation_pp import simulation_pipe
import coptpy as cp
from coptpy import COPT
from heat_simulation.simulation_device import simulation_hp,simulation_fc
import xlwt
import xlrd
import csv
from itertools import islice
import math

class MultiTime_model:
    def __init__(self, pv_lb:int, ramp_up_rate:float) -> None:
        """self 存一些MP问题的结果
        """
        self.ramp_up_rate = ramp_up_rate
        self.planning_res = None
        self.operation_res = None
        self.pv_lb = pv_lb
        pass

    def __re(self, year, life):
        l = [0 for _ in range(life)]
        for i in range(life):
            if i in [year, 2 * year, 3 * year, 4 * year, 5 * year]:
                l[i] = 1
        return l
    
    def __crf(self, year):
        i = 0.08
        crf=((1+i)**year)*i/((1+i)**year-1);
        return crf
    
    def formulate_model(self,scenario_s:object,is_SP:Boolean) -> tuple:
        """生成模型 返回一个模型object

        Args:
            scenario_s (object): _description_
            is_SP (Boolean): 

        Returns:
            tuple: 返回 模型，容量约束，平衡约束
        """
        g_demand = scenario_s.g_demand
        ele_load = scenario_s.ele_load
        water_load = scenario_s.water_load
        pv_5min = scenario_s.pv
        # pv_number = scenario_s.pv_number
        # pv_5min = pv_3[pv_number]
        days = scenario_s.days
        self.days = days
        M = 1000000
        d = 0.08  #利率
        mu_OM = 0.02 #每年运维成本系数
        
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
        cost_hst = 3000
        cost_eb = 800
        cost_pv = 5000
        cost_bs = 2300
        cost_hp = 8000
        cost_ht = 40 # 储热
        system_life = 20

        pi_EL = self.__re(10, system_life)
        pi_FC = self.__re(10, system_life)
        pi_BS = self.__re(4, system_life)


        # 系数
        k_fc = 16
        k_el = 45
        k_ghp_g = 4.4
        eta_fc = 1.2
        k_eb = 0.9
        eta_bs = 0.95
        eta_ht = 0.9
        soc_min = 0.15
        soc_max = 0.9
        bs_self_discharge = 0.01

        bs_hour = 2
        cooling_rate = 1# p=rate*c
        c = 4200 # J/kg*C
        c_kWh = 4200/3.6/1000000
        lambda_h = 200

        ## model
        env = cp.Envr()
        model = env.createModel("model")
    
        period = days * 24
        period_ele = period*12
        C_IN = model.addVar(vtype=COPT.CONTINUOUS, name = "C_IN")
        opex = model.addVar(vtype=COPT.CONTINUOUS, name = "opex")
        C_OM = model.addVar(vtype=COPT.CONTINUOUS, name = "C_OM")
        C_RE = model.addVar(vtype=COPT.CONTINUOUS, name = "C_RE")
        P_fc = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_fc")
        P_el = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_el")
        P_pv = model.addVar( lb = self.pv_lb, ub = 15000, vtype=COPT.CONTINUOUS, name = "P_pv")
        P_hp = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_hp")
        P_bs = model.addVar( lb=0, ub = 5000, vtype=COPT.CONTINUOUS, name = "P_bs")
        H_hs = model.addVar( lb=100, vtype=COPT.CONTINUOUS, name = "H_hs")
        P_eb = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_eb")
        G_ht = model.addVar( lb=0, ub = 10000, vtype=COPT.CONTINUOUS, name = "G_ht")
        # P_fl = model.addVar( lb=0, vtype=COPT.CONTINUOUS, name = "P_fl")

        # z_el = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_el")
        # z_fc = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_fc")
        g_fc_hour = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_fc_hour")
        h_fc_hour = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "h_fc_hour")
        
        h_el = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "h_el")
        p_el = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_el")
        h_hs = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "h_hs")

        p_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_hp")
        g_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_hp")
        g_rh = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_rh")
        # z_hp_g = model.addVars(period, vtype = COPT.BINARY, nameprefix = "z_hp_g")
        # z_hp_r = model.addVars(period, vtype = COPT.BINARY, nameprefix = "z_hp_r")
        # p_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_hp")
        # q_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "q_hp")
        # g_hp = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_hp")

        h_pur = model.addVars(period, lb=0, vtype = COPT.CONTINUOUS, nameprefix = "h_pur")
        #soc = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "soc") # 蓄电池

        p_eb = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "p_eb") # 电锅炉耗电
        g_eb = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_eb") # 电锅炉制热
        # 储热
        g_ht = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ht") # 储热
        g_ht_ch = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ht_ch") # 储热充电
        g_ht_di = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_ht_di") # 储热放电

        p_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_fc")
        g_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "g_fc")
        h_fc = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "h_fc")

        p_bs_di = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bs_di") # 蓄电池放电
        p_bs_ch = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bs_ch") # 蓄电池充电
        p_bs = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_bs") # 蓄电池储电量

        # z_bs_di = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_bs_di") # 蓄电池放电
        # z_bs_ch = model.addVars(period_ele, vtype = COPT.BINARY, nameprefix = "z_bs_ch") # 蓄电池充电

        p_pv = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_pv")

        # p_fl = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_fl") # 灵活负载
        # if is_SP:
        #p_us = model.addVars(period_ele, vtype = COPT.CONTINUOUS, nameprefix = "p_us") # 未满足的电负荷
        #g_us = model.addVars(period, vtype = COPT.CONTINUOUS, nameprefix = "g_us") # 未满足的热负荷、
        s_fc=model.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "s_fc")
        s_hp=model.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "s_hp")
        s_ht=model.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "s_ht")
        s_fc_cap=model.addVars(period_ele, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "s_fc_cap")
        s_hp_cap=model.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "s_hp_cap")
            
        ## constraints
        # ghp
        # model.addConstrs(g_demand[i]<=M*z_ghp_g[i] for i in range(period))
        # model.addConstrs(M*z_hp_g[i]>=g_hp[i] for i in range(period))
        # model.addConstrs(M*z_hp_r[i]>=g_rh[i] for i in range(period))
        # test = model.addConstrs(z_hp_g[i]+z_hp_r[i]<=1 for i in range(period)) # 有常数
        if not is_SP:
            model.addConstrs(p_hp[i] == g_hp[i]/k_ghp_g for i in range(period))
            
            # fuel cell
            model.addConstrs(g_fc[i] == eta_fc*k_fc*h_fc[i] for i in range(period_ele))
            model.addConstrs(p_fc[i] == k_fc*h_fc[i] for i in range(period_ele))
        model.addConstrs(g_fc[i+1] - g_fc[i] <= self.ramp_up_rate*P_fc*eta_fc for i in range(period_ele-1))
        model.addConstrs(g_fc_hour[i] == cp.quicksum(g_fc[i*12+j] for j in range(12))/12 for i in range(period))
        model.addConstrs(h_fc_hour[i] == cp.quicksum(h_fc[i*12+j] for j in range(12))/12 for i in range(period))
        # el
        model.addConstrs(p_el[i] == h_el[i]*k_el for i in range(period))
        model.addConstrs(h_el[i] <= H_hs for i in range(period))

        # model.addConstrs(M*z_fc[i] >= p_fc_hour[i] for i in range(period))
        # model.addConstrs(M*z_el[i] >= p_el[i] for i in range(period))
        # model.addConstrs(z_fc[i] + z_el[i] <= 1 for i in range(period))

        # eb
        model.addConstrs(g_eb[i] == p_eb[i]*k_eb for i in range(period))

        # pv
        model.addConstrs(p_pv[i]==pv_5min[i]*P_pv for i in range(period_ele))
  

        # balance
        #if is_SP:
        #    p_balance = model.addConstrs(p_el[i//12] + ele_load[i//12] + p_hp[i//12] + p_eb[i//12] + p_bs_ch[i] == p_us[i] + p_fc[i] + p_pv[i] + p_bs_di[i] for i in range(period_ele))
        #    g_balance = model.addConstrs(g_fc_hour[i]+g_hp[i]+g_ht_di[i]+g_eb[i] + g_us[i] == g_rh[i] + g_demand[i]+g_ht_ch[i] + water_load[i] for i in range(period))
        if not is_SP:
            g_balance = model.addConstrs(g_fc_hour[i]+g_hp[i]+g_ht_di[i]+g_eb[i] == g_rh[i] + g_demand[i]+g_ht_ch[i] + water_load[i] for i in range(period))
            p_balance = model.addConstrs(p_el[i//12] + ele_load[i//12] + p_hp[i//12] + p_eb[i//12] + p_bs_ch[i] == p_fc[i] + p_pv[i] + p_bs_di[i] for i in range(period_ele))
        model.addConstr(g_ht[0] - g_ht[period-1] == g_ht_ch[period - 1]*eta_ht - g_ht_di[period - 1]/eta_ht)
        model.addConstr(h_hs[0] - h_hs[period-1] == h_pur[period-1] + h_el[period-1] - h_fc_hour[period-1])
        model.addConstr(p_bs[0] - p_bs[period_ele-1] == p_bs_ch[period_ele-1]*eta_bs - p_bs_di[period_ele-1]/eta_bs)

        # storage
        model.addConstrs(g_ht[i+1] - g_ht[i] == g_ht_ch[i]*eta_ht - g_ht_di[i]/eta_ht for i in range(period - 1))
        model.addConstrs(h_hs[i+1] - h_hs[i] == h_pur[i] + h_el[i] - h_fc_hour[i] for i in range(period - 1))
        model.addConstrs(p_bs[i+1] - p_bs[i] == p_bs_ch[i]*eta_bs - p_bs_di[i]/eta_bs for i in range(period_ele - 1))
        

        model.addConstrs(p_bs_ch[i] <= P_bs/bs_hour for i in range(period_ele))
        model.addConstrs(p_bs_di[i] <= P_bs/bs_hour for i in range(period_ele))
        
        # model.addConstrs(M*z_bs_ch[i] >= p_bs_ch[i] for i in range(period_ele))
        # model.addConstrs(M*z_bs_di[i] >= p_bs_di[i] for i in range(period_ele))
        # model.addConstrs(z_bs_di[i] + z_bs_ch[i] <= 1 for i in range(period_ele))

        model.addConstrs(g_ht[i*24+24] == g_ht[i*24] for i in range(period//24-1))

        # model.addConstr(g_hp.sum()<=g_rh.sum()+p_hp.sum())

        if is_SP:
            model.addConstrs(p_fc[i] <= P_fc+s_fc_cap[i] for i in range(period_ele))
            model.addConstrs(p_hp[i] <= P_hp+s_hp_cap[i] for i in range(period))
        else:
            model.addConstrs(p_fc[i] <= P_fc for i in range(period_ele))
            model.addConstrs(p_hp[i] <= P_hp for i in range(period))
        model.addConstrs(p_el[i] <= P_el for i in range(period))
        model.addConstrs(p_eb[i] <= P_eb for i in range(period))
        model.addConstrs(g_ht[i] <= G_ht for i in range(period))
        model.addConstrs(h_hs[i] <= H_hs for i in range(period))
        # capactiy
        if is_SP:
            C_pv = model.addConstr(P_pv == self.planning_res['P_pv'] )
            C_fc = model.addConstr(P_fc == self.planning_res['P_fc'] )
            C_hp = model.addConstr(P_hp == self.planning_res['P_hp'] )
            C_el = model.addConstr(P_el == self.planning_res['P_el'] )
            C_eb = model.addConstr(P_eb == self.planning_res['P_eb'] )
            C_ht = model.addConstr(G_ht == self.planning_res['G_ht'] )
            C_hs = model.addConstr(H_hs == self.planning_res['H_hs'] )
            C_bs = model.addConstr(P_bs == self.planning_res['P_bs'] )
        
        # C_fl = model.addConstrs(p_fl[i] <= P_fl for i in range(period_ele))
        model.addConstrs(p_bs[i] <= soc_max * P_bs for i in range(period_ele))
        model.addConstrs(p_bs[i] >= soc_min * P_bs for i in range(period_ele))

        # obj
        model.addConstr(C_IN == cost_ht*G_ht + cost_eb*P_eb + cost_bs*P_bs + cost_pv*P_pv + cost_fc*P_fc + cost_hp*P_hp + cost_hst*H_hs + cost_el*P_el)
        model.addConstr(C_OM == cp.quicksum(mu_OM * C_IN / pow((1 + d), (i + 1)) for i in range(system_life)))
        model.addConstr(C_RE == cp.quicksum(
            (pi_EL[i] * cost_el * P_el + pi_FC[i] * cost_fc * P_fc + pi_BS[i] * cost_bs * P_bs) / pow((1 + d), (i + 1)) for i in
            range(system_life)))
        model.addConstr(opex == h_pur.sum()*lambda_h)
        if is_SP:
            model.setObjective(s_fc.sum()+s_hp.sum()+s_ht.sum()+s_fc_cap.sum()+s_hp_cap.sum() , sense=COPT.MINIMIZE)
        else:
            model.setObjective(self.__crf(system_life) * (C_IN + C_OM + C_RE) + opex, sense=COPT.MINIMIZE)

        cap_constraints_dict = {
            'C_pv':C_pv,
            'C_fc':C_fc,
            'C_hp':C_hp,
            'C_el':C_el,
            'C_eb':C_eb,
            'C_ht':C_ht,
            'C_hs':C_hs,
            'C_bs':C_bs,
            # 'test':test,
        } if is_SP else None

        balance_constraints_dict = {
            "g_balance":0,#g_balance,
            "p_balance":0#p_balance,
        } if is_SP else None
        return model,cap_constraints_dict,balance_constraints_dict
    
    def GHE(self):
        #导入g函数
        f = csv.reader(open('gFunYear2.csv', 'r'))
        gFun2 = []
        for l in islice(f, 0, None):
            gFun2.append(float(l[1]))

        #地埋管相关参数
        cpw=4.2 #kW/(kg.K)
        m=0.3833
        T_g_initial=15
        kf=0.3387167691700763
        n=200
        H=192
        kai=2.07

        month=30 #一个月30天
        day=24 #一天24小时
        year=12 #一年12个月
        period=self.days*day #总共288小时

        g_ghe=self.MP.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "g_ghe")
        g_ghe_mean=self.MP.addVars(self.days, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "g_ghe_mean")

        T_b_start=self.MP.addVars(self.days, lb=-COPT.INFINITY,vtype = COPT.CONTINUOUS, nameprefix = "T_b_start")

        T_b_first=self.MP.addVars(period, lb=-COPT.INFINITY,vtype = COPT.CONTINUOUS, nameprefix = "T_b_first")
        T_in_first=self.MP.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "T_in_first")
        T_out_first=self.MP.addVars(period, lb=4,vtype = COPT.CONTINUOUS, nameprefix = "T_out_first")

        T_b_last=self.MP.addVars(period, lb=-COPT.INFINITY,vtype = COPT.CONTINUOUS, nameprefix = "T_b_last")
        T_in_last=self.MP.addVars(period, lb=0,vtype = COPT.CONTINUOUS, nameprefix = "T_in_last")
        T_out_last=self.MP.addVars(period, lb=4,vtype = COPT.CONTINUOUS, nameprefix = "T_out_last")

        g_hp = [self.MP.getVarByName(f'g_hp({i})') for i in range(period)]
        p_hp = [self.MP.getVarByName(f'p_hp({i})') for i in range(period)]

        self.MP.addConstrs(g_ghe[i] == g_hp[i]-p_hp[i] for i in range(period))

        for i in range(self.days):
            self.MP.addConstr(g_ghe_mean[i]==sum([g_ghe[day*i+j] for j in range(day)])/day)

            if i==0:
                self.MP.addConstr(T_b_start[i]==T_g_initial)
            else:
                self.MP.addConstr(T_b_start[i]==T_b_start[i-1]-1000*g_ghe_mean[i-1]*gFun2[month*day]/(2*math.pi*kai*n*H))
        
            for k in range(day):
                self.MP.addConstr(T_b_first[day*i+k]==T_b_start[i]-1000*sum([g_ghe[day*i+l]*(gFun2[l+1]-gFun2[l])  for l in range(k)])/(2*math.pi*kai*n*H))
                self.MP.addConstr(T_b_last[day*i+k]==T_b_start[i]-1000*g_ghe_mean[i]*gFun2[(month-1)*day]/(2*math.pi*kai*n*H)-1000*sum([g_ghe[day*i+l]*(gFun2[l+1]-gFun2[l])  for l in range(k)])/(2*math.pi*kai*n*H))

                self.MP.addConstr(T_out_first[day*i+k]==kf*(T_in_first[day*i+k]-T_b_first[day*i+k])+T_b_first[day*i+k])
                self.MP.addConstr(T_out_last[day*i+k]==kf*(T_in_last[day*i+k]-T_b_last[day*i+k])+T_b_last[day*i+k])

                self.MP.addConstr(g_ghe[day*i+k]==n*cpw*m*(T_out_first[day*i+k]-T_in_first[day*i+k]))
                self.MP.addConstr(g_ghe[day*i+k]==n*cpw*m*(T_out_last[day*i+k]-T_in_last[day*i+k]))
        
        return None
        

    def formulate_MP(self,scenario_s:object) -> None:
        self.MP_days = scenario_s.days
        self.MP_g_demand = scenario_s.g_demand
        self.MP_water_load = scenario_s.water_load
        self.MP_ele_load = scenario_s.ele_load
        self.MP,_,_ = self.formulate_model(scenario_s, is_SP=False)
        # self.GHE()
    
    @timer
    def solve_MP(self,wb,sheet,num=0) -> None:
        """主要是更新self.planning_res 用于增加cut
        """
        self.MP.setParam(COPT.Param.RelGap, 0.001)
        self.MP.solve()
        if self.MP.status == 2:
            self.MP.computeIIS() 
            self.MP.writeIIS("example.iis")
        period = 24
        self.planning_res = {
            'P_fc':self.MP.getVarByName('P_fc').x,
            'P_el':self.MP.getVarByName('P_el').x,
            'P_pv':self.MP.getVarByName('P_pv').x,
            'P_hp':self.MP.getVarByName('P_hp').x,
            'P_bs':self.MP.getVarByName('P_bs').x,
            'H_hs':self.MP.getVarByName('H_hs').x,
            'P_eb':self.MP.getVarByName('P_eb').x,
            'G_ht':self.MP.getVarByName('G_ht').x,
        }

        items = list(self.planning_res.keys())
        for i in range(len(items)):
            sheet.write(num+1, i, self.planning_res[items[i]])
        filename='容量记录'+'.xls'
        wb.save(filename)


        self.operation_res = [{
            'g_fc':[self.MP.getVarByName(f'g_fc_hour({24*d + i})').x for i in range(period)],
            'h_fc':[self.MP.getVarByName(f'h_fc({24*d + i})').x for i in range(period)],
            'g_hp':[self.MP.getVarByName(f'g_hp({24*d + i})').x for i in range(period)],
            'p_hp':[self.MP.getVarByName(f'p_hp({24*d + i})').x for i in range(period)],
            'g_ht_di':[self.MP.getVarByName(f'g_ht_di({24*d + i})').x for i in range(period)],
            'g_ht_ch':[self.MP.getVarByName(f'g_ht_ch({24*d + i})').x for i in range(period)],
            'g_eb':[self.MP.getVarByName(f'g_eb({24*d + i})').x for i in range(period)],
            'g_total_demand':[self.MP_water_load[24*d + i] + self.MP_g_demand[24*d + i] for i in range(period)],
            'g_rh':[self.MP.getVarByName(f'g_rh({24*d + i})').x for i in range(period)],
        } for d in range(self.MP_days)]
        self.operation_res_5min = [{
            'p_fc':[self.MP.getVarByName(f'p_fc({12*d + i})').x for i in range(period*12)],
        } for d in range(self.MP_days)]
        1

    def get_MP_result(self,num=0) -> None:

        period = self.MP_days * 24
        period_ele = self.MP_days * 24 * 12
        # allvars = self.MP.getVars()
        # for var in allvars:
        #     print(" x[{0}]: {1}".format(var.getName(), var.x))
        res_5min = {
            
            "ele_load":[self.MP_ele_load[i//12] for i in range(period_ele)],
            "p_el": [self.MP.getVarByName(f'p_el({i//12})').x for i in range(period_ele)],
            "p_hp": [self.MP.getVarByName(f'p_hp({i//12})').x for i in range(period_ele)],
            "p_eb": [self.MP.getVarByName(f'p_eb({i//12})').x for i in range(period_ele)],

            "p_pv": [self.MP.getVarByName(f'p_pv({i})').x for i in range(period_ele)],
            "p_fc": [self.MP.getVarByName(f'p_fc({i})').x for i in range(period_ele)],

            "p_bs": [self.MP.getVarByName(f'p_bs({i})').x for i in range(period_ele)],
            "p_bs_ch": [self.MP.getVarByName(f'p_bs_ch({i})').x for i in range(period_ele)],
            "p_bs_di": [self.MP.getVarByName(f'p_bs_di({i})').x for i in range(period_ele)],
        }

        res_hour = {
            "ele_load": self.MP_ele_load,
            "p_el": [self.MP.getVarByName(f'p_el({i})').x for i in range(period)],
            "p_hp": [self.MP.getVarByName(f'p_hp({i})').x for i in range(period)],
            "p_eb": [self.MP.getVarByName(f'p_eb({i})').x for i in range(period)],

            "p_pv_hour": [sum([self.MP.getVarByName(f'p_pv({12*i+j})').x for j in range(12)])/12 for i in range(period)],
            "p_fc_hour": [sum([self.MP.getVarByName(f'p_fc({12*i+j})').x for j in range(12)])/12 for i in range(period)],

            "p_bs_hour": [sum([self.MP.getVarByName(f'p_bs({12*i+j})').x for j in range(12)])/12 for i in range(period)],
            "p_bs_ch_hour":  [sum([self.MP.getVarByName(f'p_bs_ch({12*i+j})').x for j in range(12)])/12 for i in range(period)],
            "p_bs_di_hour":  [sum([self.MP.getVarByName(f'p_bs_di({12*i+j})').x for j in range(12)])/12 for i in range(period)],
            #"p_us_hour": [sum([self.MP.getVarByName(f'p_us({12*i+j})').x for j in range(12)])/12 for i in range(period)],

            "g_ht": [self.MP.getVarByName(f'g_ht({i})').x for i in range(period)],
            "g_ht_ch": [self.MP.getVarByName(f'g_ht_ch({i})').x for i in range(period)],
            "g_ht_di": [self.MP.getVarByName(f'g_ht_di({i})').x for i in range(period)],
            "g_hp": [self.MP.getVarByName(f'g_hp({i})').x for i in range(period)],
            "g_eb":[self.MP.getVarByName(f'g_eb({i})').x for i in range(period)],
            "g_fc_hour": [self.MP.getVarByName(f'g_fc_hour({i})').x for i in range(period)],
            "g_demand": self.MP_g_demand,
            "water_load": self.MP_water_load,
            "total_g_load":[self.MP_g_demand[i] + self.MP_water_load[i] for i in range(period)],
            
            "g_rh":[self.MP.getVarByName(f'g_rh({i})').x for i in range(period)],
            #"g_us_hour": [sum([self.MP.getVarByName(f'p_us({12*i+j})').x for j in range(12)])/12 for i in range(period)],

            "h_fc_hour": [self.MP.getVarByName(f'h_fc_hour({i})').x for i in range(period)],
            "h_el": [self.MP.getVarByName(f'h_el({i})').x for i in range(period)],
            "h_hs": [self.MP.getVarByName(f'h_hs({i})').x for i in range(period)],
            "h_pur": [self.MP.getVarByName(f'h_pur({i})').x for i in range(period)],
        }
        res_obj = {
            "obj": self.MP.objVal,
            "C_IN": self.MP.getVarByName('C_IN').x,
            "C_OM": self.MP.getVarByName('C_OM').x,
            "C_RE": self.MP.getVarByName('C_RE').x,
            "opex": self.MP.getVarByName('opex').x,
        }
        pd.DataFrame(res_5min).to_csv("ramp"+str(int(1/self.ramp_up_rate))+"-5min.csv")
        pd.DataFrame(res_hour).to_csv("ramp"+str(int(1/self.ramp_up_rate))+"-hour.csv")
        pd.DataFrame(res_obj,index=[0]).to_csv("ramp"+str(int(1/self.ramp_up_rate))+"-obj.csv")
        pd.DataFrame(self.planning_res,index=[0]).to_csv("ramp"+str(int(1/self.ramp_up_rate))+"-devicecap.csv")
        pprint.pprint(self.planning_res)

    

    def simulation_heat(self,operation_res:list,operation_res_5min:list,scenario_s:object,operation_revise:dict,num:int,iter=0) -> dict:
        """在子问题中调用，热传输过程仿真，返回每一时段每个设备的能量违反度。序贯仿真fc和hp的设备

        Args:
            operation_res (dict): !得加这个因为
            scenario_s (object): _description_
            operation_revise (dict): revised operation

        Returns:
            dict: 运行结果，供热设备的约束违反量
        """

        T_dict,mass_dict = simulation_pipe(operation_res,scenario_s,operation_revise)
        cop_hp = simulation_hp(T_dict['hp'])
        cop_fc, Electrothermal = simulation_fc(self.planning_res['P_fc'], operation_res_5min['p_fc'])
        # 每个设备温度界限参数
        T_hp_max,T_hp_min = 55, 45
        T_fc_max,T_fc_min = 65, 45
        T_ht_max,T_ht_min = 60, 45
        # T_rh_max,T_rh_min = 60, 45
        c_kWh = 4200/3.6/1000000
        # g_revise = lambda t_max,t_min,t,m: 0 if t_min<=t<=t_max else (c_kWh*m*(t-t_max) if t>t_max else c_kWh*m*(t-t_min))
        operation_revise = {
            'fail':False,
            # 'g_fc':[c_kWh*1000*mass_dict['fc'][i]*(T_dict['fc'][i]-T_fc_max) if T_dict['fc'][i]>T_fc_max else 0 for i in range(len(T_dict['fc']))],
            # 'g_hp':[c_kWh*1000*mass_dict['hp'][i]*(T_dict['hp'][i]-T_hp_max) if T_dict['hp'][i]>T_hp_max else 0 for i in range(len(T_dict['hp']))],
            # 'g_ht':[c_kWh*1000*mass_dict['ht'][i]*(T_dict['ht'][i]-T_ht_max) if T_dict['ht'][i]>T_ht_max else 0 for i in range(len(T_dict['ht']))],
            'g_fc':[c_kWh*1000*mass_dict['fc'][i]*(T_dict['fc'][i]-T_fc_max) for i in range(len(T_dict['fc']))],
            'g_hp':[c_kWh*1000*mass_dict['hp'][i]*(T_dict['hp'][i]-T_hp_max) for i in range(len(T_dict['hp']))],
            'g_ht':[c_kWh*1000*mass_dict['ht'][i]*(T_dict['ht'][i]-T_ht_max) for i in range(len(T_dict['ht']))],
            'cop_fc':cop_fc,
            'Electrothermal':Electrothermal,
            'cop_hp':cop_hp-1, # 弄低一点
            # 'g_rh':[c_kWh*1000*mass_dict['rh'][i]*(T_dict['rh'][i]-T_rh_max) if T_dict['rh'][i]>T_rh_max else 0 for i in range(len(T_dict['rh']))],
        }

        data_recording={
            'T_fc':T_dict['fc'],
            'T_hp':T_dict['hp'],
            'T_ht':T_dict['ht'],
            'm_fc':mass_dict['fc'],
            'm_hp':mass_dict['hp'],
            'm_ht':mass_dict['ht'],
            'g_fc':operation_res['g_fc'],
            'g_hp':operation_res['g_hp'],
            'cop_fc':cop_fc,
            'cop_hp':cop_hp,
            'g_fc_revise':operation_revise['g_fc'],
            'g_hp_revise':operation_revise['g_hp'],
            'g_ht_revise':operation_revise['g_ht'],
        }
        wb = xlwt.Workbook()
        crucial_data = wb.add_sheet('温度、能效比')
        items = list(data_recording.keys())
        for i in range(len(items)):
            crucial_data.write(0, i, items[i])
            if type(data_recording[items[i]]) == list or type(data_recording[items[i]])==type(T_dict['fc']):
                for j in range(len(data_recording[items[i]])):
                    crucial_data.write(j+1, i, (data_recording[items[i]])[j])
            else:
                crucial_data.write(1, i, data_recording[items[i]])
        if iter==0:
            filename = '原始温度、能效比记录_场景'+str(num)+ '.xls'
        else:
            filename = '温度、能效比记录_迭代_场景'+str(num)+'.xls'
        wb.save(filename)
        

        operation_revise['fail'] = True if max(operation_revise['g_fc'])>0 or max(operation_revise['g_hp'])>0 or max(operation_revise['g_ht'])>0 else False
        
        #print('\n'+'超出安全范围：'+str(max(operation_revise['g_fc']))+'  '+str(max(operation_revise['g_hp']))+'  '+str(max(operation_revise['g_ht']))+'\n')
        
        return operation_revise
        
    
    def SP_result_debug(self,SP:object,scenario_s:object):
        period_ele= 24*12
        period = 24

        res_5min = [{
            'p_fc':[SP.getVarByName(f'p_fc({12*d + i})').x for i in range(period_ele)],
        } for d in range(scenario_s.days)]
        res_hour = [{
            'g_fc':[SP.getVarByName(f'g_fc_hour({24*d + i})').x for i in range(period)],
            'h_fc':[SP.getVarByName(f'h_fc({24*d + i})').x for i in range(period)],
            'g_hp':[SP.getVarByName(f'g_hp({24*d + i})').x for i in range(period)],
            'p_hp':[SP.getVarByName(f'p_hp({24*d + i})').x for i in range(period)],
            'g_ht_di':[SP.getVarByName(f'g_ht_di({24*d + i})').x for i in range(period)],
            'g_ht_ch':[SP.getVarByName(f'g_ht_ch({24*d + i})').x for i in range(period)],
            'g_eb':[SP.getVarByName(f'g_eb({24*d + i})').x for i in range(period)],
            'g_total_demand':[scenario_s.water_load[24*d + i] + scenario_s.g_demand[24*d + i] for i in range(period)],
            'g_rh':[SP.getVarByName(f'g_rh({24*d + i})').x for i in range(period)],
        } for d in range(scenario_s.days)]

        pd.DataFrame(res_hour).to_csv("SP_debug_hour.csv")

        return [res_5min,res_hour]

   
    def add_cut(self,operation_res:list,scenario_s:object, operation_revise:dict,num) :
        # 每次调用add cut 构建一个子问题，cut加到self的MP里面
        period = scenario_s.days * 24
        SP,sp_dual,balance_dual = self.formulate_model(scenario_s, is_SP = True)
        g_fc_hour = [SP.getVarByName(f'g_fc_hour({i})') for i in range(period)]
        g_fc = [SP.getVarByName(f'g_fc({i})') for i in range(period*12)]
        p_fc = [SP.getVarByName(f'p_fc({i})') for i in range(period*12)]
        h_fc = [SP.getVarByName(f'h_fc({i})') for i in range(period*12)]
        g_hp = [SP.getVarByName(f'g_hp({i})') for i in range(period)]
        p_hp = [SP.getVarByName(f'p_hp({i})') for i in range(period)]
        g_rh = [SP.getVarByName(f'g_rh({i})') for i in range(period)]
        g_ht_ch = [SP.getVarByName(f'g_ht_ch({i})') for i in range(period)]
        g_ht_di = [SP.getVarByName(f'g_ht_di({i})') for i in range(period)]
        s_fc= [SP.getVarByName(f's_fc({i})') for i in range(period)]
        s_hp= [SP.getVarByName(f's_hp({i})') for i in range(period)]
        s_ht= [SP.getVarByName(f's_ht({i})') for i in range(period)]

        ## 仿真的约束
        # for i in range(period):
        #     if operation_revise['g_fc'][i] != 0:
        #         SP.addConstr(g_fc[i] <= operation_res['g_fc'][i] - operation_revise['g_fc'][i])
        #     if operation_revise['g_hp'][i] != 0:
        #         SP.addConstr(g_hp[i] <= operation_res['g_hp'][i] - operation_revise['g_hp'][i])
        #     if operation_revise['g_ht'][i] != 0:
        #         SP.addConstr(g_ht_di[i] <= max(0, operation_res['g_ht_di'][i] - operation_revise['g_ht'][i]))
        cons1 = SP.addConstrs(g_fc_hour[i] <= operation_res['g_fc'][i] - operation_revise['g_fc'][i]+s_fc[i] for i in range(period))
        cons2 = SP.addConstrs(g_hp[i] <= operation_res['g_hp'][i] - operation_revise['g_hp'][i]+s_hp[i] for i in range(period))
        # SP.addConstrs(g_rh[i] <= operation_res['g_rh'][i] - operation_revise['g_rh'][i] for i in range(period))
        cons3 = SP.addConstrs(g_ht_di[i] <= max(0, operation_res['g_ht_di'][i] - operation_revise['g_ht'][i])+s_ht[i] for i in range(period))
        
        SP.addConstrs(p_hp[i] == g_hp[i]/operation_revise['cop_hp'][i] for i in range(period))        
        SP.addConstrs(g_fc[i] == operation_revise['Electrothermal'][i]*p_fc[i] for i in range(period*12))
        SP.addConstrs(p_fc[i] == operation_revise['cop_fc'][i]*h_fc[i] for i in range(period*12))

        cons1=[]
        cons2=[]
        cons3=[]
        for i in range(period):
            con1=SP.addConstr(g_fc_hour[i]==operation_res['g_fc'][i])
            con2=SP.addConstr(g_hp[i]==operation_res['g_hp'][i])
            con3=SP.addConstr(g_ht_di[i]==operation_res['g_ht_di'][i])
            cons1.append(con1)
            cons2.append(con2)
            cons3.append(con3)
        ## 约束的对偶,增加cut
        # SP.setParam(COPT.Param.Dualize,1)
        # 求解线性模型，获得对偶值
        SP.solveLP()
        if SP.status==COPT.INFEASIBLE:
            SP.computeIIS()
            SP.writeIIS('opmodel.ilp')

        #print('\n'+'当前误差为'+str(SP.objVal)+'\n')

        feasible_SP = 1 if SP.objVal == 0 else 0
        if feasible_SP:
            return SP.objVal
        self.SP_result_debug(SP,scenario_s)
        
        g_fc_MP=[self.MP.getVarByName(f'g_fc_hour({i+num*24})') for i in range(period)]
        g_hp_MP=[self.MP.getVarByName(f'g_hp({i+num*24})') for i in range(period)]
        g_ht_MP=[self.MP.getVarByName(f'g_ht_di({i+num*24})') for i in range(period)]
        1
        P_pv = self.MP.getVarByName('P_pv')
        P_fc = self.MP.getVarByName('P_fc')
        P_hp = self.MP.getVarByName('P_hp')
        P_el = self.MP.getVarByName('P_el')
        P_eb = self.MP.getVarByName('P_eb')
        G_ht = self.MP.getVarByName('G_ht')
        H_hs = self.MP.getVarByName('H_hs')
        P_bs = self.MP.getVarByName('P_bs')
        # for n,c in sp_dual.items():
        #     print(n,c.pi)
        
        for i in range(period):
            self.MP.addConstr(s_fc[i].x+s_hp[i].x+s_ht[i].x+cons1[i].pi*(g_fc_MP[i]-operation_res['g_fc'][i])+cons2[i].pi*(g_hp_MP[i]-operation_res['g_hp'][i])
            +cons3[i].pi*(g_ht_MP[i]-operation_res['g_ht_di'][i])<=0)

        #self.MP.addConstr(SP.objVal 
        #                  +sum([cons1[i].pi*(g_fc_MP[i]-operation_res['g_fc'][i])+cons2[i].pi*(g_hp_MP[i]-operation_res['g_hp'][i])
        #                    +cons3[i].pi*(g_ht_MP[i]-operation_res['g_ht_di'][i]) for i in range(period)])
        #                #   - cp.quicksum(balance_dual['g_balance'][i].pi*SP.getVarByName(f'g_us({i})').x for i in range(period))
        #                #   - cp.quicksum(balance_dual['p_balance'][i].pi*SP.getVarByName(f'p_us({i})').x for i in range(period*12))
        #                  + sp_dual['C_pv'].pi*(P_pv - SP.getVarByName('P_pv').x) 
        #                  + sp_dual['C_fc'].pi*(P_fc - SP.getVarByName('P_fc').x) 
        #                  + sp_dual['C_hp'].pi*(P_hp - SP.getVarByName('P_hp').x) 
        #                  + sp_dual['C_el'].pi*(P_el - SP.getVarByName('P_el').x) 
        #                  + sp_dual['C_eb'].pi*(P_eb - SP.getVarByName('P_eb').x) 
        #                  + sp_dual['C_ht'].pi*(G_ht - SP.getVarByName('G_ht').x) 
        #                  + sp_dual['C_hs'].pi*(H_hs - SP.getVarByName('H_hs').x) 
        #                  + sp_dual['C_bs'].pi*(P_bs - SP.getVarByName('P_bs').x) <= 0)
        return SP.objVal
        #1 test @gwyxjtu

if __name__ == "__main__":
    main_model = MultiTime_model(1/6) # 提升30min
    
    