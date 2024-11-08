'''
Author: working-guo 867718012@qq.com
Date: 2023-07-18 16:06:18
LastEditors: guo-4060ti 867718012@qq.com
LastEditTime: 2024-11-08 10:40:43
FilePath: /copt_multi-time/model_HIES/scenario_calss.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.
Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
class Scenario:
    def __init__(self,g_demand:list,ele_load:list,water_load:list,pv:list,days:int) -> None:
        self.g_demand = g_demand
        self.ele_load = ele_load
        self.water_load = water_load
        self.pv = pv
        self.days = days


class Scenario_all_day:
    """场景定义为，热需求，电需求，生活热水需求，光照强度。大场景里面有子场景。
    """
    def __init__(self,days:int,pv_number:int,g_demand:list, ele_load:list, water_load:list, pv_3:list) -> None:
        """初始化场景，

        Args:
            days (int):场景内包含的天数
            pv_number (int):选择光伏的数目
            g_demand (list): 热负荷二维数组
            ele_load (list): _description_
            water_load (list): _description_
            pv_3 (list): _description_
        """
        self.g_demand = g_demand[:days]
        self.ele_load = ele_load[:days]
        self.water_load = water_load[:days]
        self.pv_3 = pv_3
        self.pv_number = pv_number
        self.days = days

    @property
    def main_scenario(self)->tuple:
        """总场景，用于给主问题提供数据

        Returns:
            tuple: 热负荷、电负荷、热水负荷、光伏强度、时间周期
        """
        g_demand = [item for sublist in self.g_demand for item in sublist] if self.days>1 else self.g_demand
        ele_load = [item for sublist in self.ele_load for item in sublist] if self.days>1 else self.ele_load
        water_load = [item for sublist in self.water_load for item in sublist] if self.days>1 else self.water_load
        pv = [item for sublist in self.pv_3[self.pv_number] for item in sublist] if self.days>1 else self.pv_3[self.pv_number]
        return Scenario(g_demand,ele_load,water_load,pv,self.days)
    
    @property
    def sub_scenario(self)->list:
        """长度为days的子场景

        Returns:
            list: 每个list元素是一个g_demand,ele_load,water_load,pv,period的scenario class
        """
        return [Scenario(self.g_demand[d],self.ele_load[d],self.water_load[d],self.pv_3[self.pv_number][d],1)  for d in range(self.days)]
    