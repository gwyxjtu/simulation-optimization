'''
Author: working-guo 867718012@qq.com
Date: 2023-07-19 10:35:01
LastEditors: working-guo 867718012@qq.com
LastEditTime: 2023-07-25 09:35:43
FilePath: \copt_multi-time\heat_simulation\simulation_pp.py
Description: 人一生会遇到约2920万人,两个人相爱的概率是0.000049,所以你不爱我,我不怪你.
Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import pandapipes as pp
import pprint
import pandapipes.plotting as plot
from pandapipes.plotting.simple_plot import simple_plot as sp
import pandas as pd
import pandapower.control as control
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapipes.timeseries import run_timeseries

def construct_network() -> object:
    net = pp.create_empty_network(fluid ="water")

    j1 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 1")
    j2 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 2")
    j3 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 3")# t_fluid_k 是初始的温度
    j4 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 4")
    j5 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 5")
    j6 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 6")
    j7 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 7")
    j8 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 8")
    j9 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 9")
    j10 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 10")
    j11 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 11")
    j12 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 12")
    j13 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 13")
    j14 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 14")

    j15 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 15")
    j16 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 16")
    j17 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 17")
    pp.create_circ_pump_const_pressure(net, j9, j10, p_flow_bar=7, plift_bar=5,
                                    t_flow_k=273.55+50, type="auto")

    # create heat exchanger
    pp.create_heat_exchanger(net, from_junction=j3, to_junction=j6, diameter_m=100e-3, qext_w = -1500000)#供热
    pp.create_heat_exchanger(net, from_junction=j4, to_junction=j7, diameter_m=100e-3, qext_w = -500000)
    pp.create_heat_exchanger(net, from_junction=j5, to_junction=j8, diameter_m=100e-3, qext_w = -1000000)
    pp.create_heat_exchanger(net, from_junction=j10, to_junction=j11, diameter_m=400e-3, qext_w = 3000000)
    pp.create_heat_exchanger(net, from_junction=j12, to_junction=j13, diameter_m=200e-3, qext_w = 3000000)

    # pp.create_valve(net, j6, j15, opened = True, diameter_m=2e-3, name="valve1")
    # pp.create_valve(net, j7, j16, opened = True, diameter_m=2e-3, name="valve2")
    # pp.create_valve(net, j8, j17, opened = True, diameter_m=2e-3, name="valve3")
    # pp.create_valve(net, j10, j12, opened = True, diameter_m=2e-3, name="valve4")
    pp.create_flow_control(net, j6, j15, controlled_mdot_kg_per_s = 0.1, diameter_m = 2e-3, control_active = True)
    pp.create_flow_control(net, j7, j16, controlled_mdot_kg_per_s = 0.1, diameter_m = 2e-3, control_active = True)
    pp.create_flow_control(net, j8, j17, controlled_mdot_kg_per_s = 0.1, diameter_m = 2e-3, control_active = True)
    pp.create_flow_control(net, j10, j12, controlled_mdot_kg_per_s = 0.1, diameter_m = 2e-3, control_active = True)
    # create pipe
    pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1,
                                diameter_m=150e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=293.15)

    pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.1,
                                diameter_m=100e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=293.15)
    pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=0.2,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=293.15)
    pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j5, length_km=0.2,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=293.15)
    
    pp.create_pipe_from_parameters(net, from_junction=j6, to_junction=j15, length_km=0.01,
                                diameter_m=100e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    pp.create_pipe_from_parameters(net, from_junction=j7, to_junction=j16, length_km=0.02,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    pp.create_pipe_from_parameters(net, from_junction=j8, to_junction=j17, length_km=0.08,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    pp.create_pipe_from_parameters(net, from_junction=j10, to_junction=j12, length_km=0.08,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    

    pp.create_pipe_from_parameters(net, from_junction=j15, to_junction=j9, length_km=0.01,
                                diameter_m=100e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    pp.create_pipe_from_parameters(net, from_junction=j16, to_junction=j9, length_km=0.02,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    pp.create_pipe_from_parameters(net, from_junction=j17, to_junction=j9, length_km=0.08,
                                diameter_m=250e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 3, text_k=283)
    # pp.create_pipe_from_parameters(net, from_junction=j10, to_junction=j12, length_km=0.58,
    #                             diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 5, text_k=283)

    pp.create_pipe_from_parameters(net, from_junction=j11, to_junction=j1, length_km=1.08,
                                diameter_m=400e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 10, text_k=293.15)
    pp.create_pipe_from_parameters(net, from_junction=j13, to_junction=j1, length_km=0.58,
                                diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=11.63, sections = 5, text_k=293.15)

    pp.create_sink(net, junction=j14, mdot_kg_per_s=0.0, name="Sink")
    pp.create_source(net, junction=j14, mdot_kg_per_s=0.0, name="Source 1")

    return net
def ow_print(net, period):
    log_variables = [('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
                    ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
                    ('res_pipe', 't_from_k'), ('res_pipe', 't_to_k'),('res_pipe', 'mdot_from_kg_per_s'),
                    ('res_heat_exchanger', 't_from_k'), ('res_heat_exchanger', 't_to_k'), ('res_heat_exchanger', 'mdot_from_kg_per_s'),
                    ('res_ext_grid', 'mdot_kg_per_s')]
    operation_res = [('res_heat_exchanger', 't_from_k'), 
                     ('res_heat_exchanger', 't_to_k'), 
                     ('res_heat_exchanger', 'mdot_from_kg_per_s'),]
    ow = OutputWriter(net, range(period), output_path=None, log_variables=log_variables)
    return ow
def print_res(net,ow):
    # res_pipe
    print(net.res_pipe)
    # res_junction
    print(net.res_junction)
    # res_heat_exchanger
    print(net.res_heat_exchanger)
    # res_circ_pump
    # print(net.res_circ_pump_pressure)
    print("pipe:")
    print(ow.np_results["res_pipe.t_from_k"])
    print(ow.np_results["res_pipe.t_to_k"])
    print(ow.np_results["res_pipe.mdot_from_kg_per_s"])

    print("res_heat_exchanger:")
    print("t_from_k:")
    print(ow.np_results["res_heat_exchanger.t_from_k"])
    print("t_to_k:")
    print(ow.np_results["res_heat_exchanger.t_to_k"])
    print("mdot_from_kg_per_s:")
    print(ow.np_results["res_heat_exchanger.mdot_from_kg_per_s"])

def simulation_pipe(operation_res:dict,scenario_s:object) -> dict:
    """主仿真函数

    Args:
        operation_res (dict): 运行工况，主要是供热设备的运行功率，
        scenario_s (object): _description_

    Returns:
        dict: _description_
    """
    period = scenario_s.days * 24
    
    heat_exchanger_control = [[-operation_res['g_fc'][i] for i in range(period)], 
                                [-operation_res['g_hp'][i] for i in range(period)],
                                [operation_res['g_ht_ch'][i]-operation_res['g_ht_di'][i] for i in range(period)],
                                operation_res['r_hp'],
                                operation_res['g_total_demand']]
    # heat_exchanger_control = [[0 for i in range(period)],
    #                             [0 for i in range(period)],
    #                             [0 for i in range(period)],
    #                             [0 for i in range(period)],
    #                             [0 for i in range(period)]]
    valve_control = [[True if i > 0 else False for i in operation_res['g_fc']],
                     [True if i > 0 else False for i in operation_res['g_hp']],
                     [True if i-j != 0 else False for i in operation_res['g_ht_ch'] for j in operation_res['g_ht_ch']],
                     [True if i > 0 else False for i in operation_res['r_hp']],]
                    #  [True if i > 0 else False for i in operation_res['g_total_demand']]]
    valve_control = [[False  for _ in operation_res['g_fc']],
                        [False  for _ in operation_res['g_hp']], 
                        [False  for _ in operation_res['g_ht_ch']],
                        [True  for _ in operation_res['r_hp']],]
    
    net = construct_network()
    control.ConstControl(net, element='heat_exchanger', variable='qext_w',
                                  element_index=net.heat_exchanger.index.values, 
                                  data_source=DFData(pd.DataFrame(list(map(list, zip(*heat_exchanger_control))), columns=net.heat_exchanger.index.values.astype(int))),
                                  profile_name=net.heat_exchanger.index.values.astype(int))
    # control.ConstControl(net, element='valve', variable='opened',
    #                               element_index=net.valve.index.values, 
    #                               data_source=DFData(pd.DataFrame(list(map(list, zip(*valve_control))), columns=net.valve.index.values.astype(str))),
    #                               profile_name=net.valve.index.values.astype(str))
    control.ConstControl(net, element='flow_control', variable='in_service ',
                                  element_index=net.flow_control.index.values, 
                                  data_source=DFData(pd.DataFrame(list(map(list, zip(*valve_control))), columns=net.flow_control.index.values.astype(int))),
                                  profile_name=net.flow_control.index.values.astype(int))
    # control.ConstControl(net, element='flow_control', variable='control_active',
    #                               element_index=net.flow_control.index.values, 
    #                               data_source=DFData(pd.DataFrame(list(map(list, zip(*valve_control))), columns=net.flow_control.index.values.astype(str))),
    #                               profile_name=net.flow_control.index.values.astype(str))
    ow = ow_print(net,period)
    run_timeseries(net, range(period), mode='all') # hydraulics
    print_res(net,ow)
    pass