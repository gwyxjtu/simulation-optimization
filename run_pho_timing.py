'''PHO timing/memory runner for SMC revision R2.8.

Usage:
  python -u run_pho_timing.py --days 4 --smoke
  python -u run_pho_timing.py --days 12 --max-iter 20
  python -u run_pho_timing.py --days 48 --max-iter 20
'''
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import warnings
from datetime import datetime

import pandas as pd
import xlwt

from model_HIES.model_class import MultiTime_model
from model_HIES.model_load_day import SAMPLE_SEED, get_load
from model_HIES.scenario_calss import Scenario_all_day

warnings.filterwarnings("ignore")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def sim_excess_kwh(revise: dict) -> float:
    excess = 0.0
    for key in ("g_fc", "g_hp", "g_ht"):
        excess += float(sum(max(v, 0.0) for v in revise[key]))
    return excess


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(n_days: int, max_iter: int, smoke: bool, seed: int) -> None:
    os.makedirs("res", exist_ok=True)
    tag = f"d{n_days}_{'smoke' if smoke else 'pho'}"
    timing_path = f"res/pho_timing_{tag}.csv"
    days_path = f"res/pho_scenario_days_{tag}.json"

    g_demand, ele_load, water_load, pv_3, scenario_days = get_load(n_days=n_days, seed=seed)
    with open(days_path, "w") as f:
        json.dump(
            {"n_days": n_days, "seed": seed, "scenario_days": scenario_days, "nested_note": "same seed => 4 subset 12 subset 48"},
            f,
            indent=2,
        )
    log(f"sampled {len(scenario_days)} days: {scenario_days[:8]}...")

    all_scenario = Scenario_all_day(n_days, 1, g_demand, ele_load, water_load, pv_3)
    assert all_scenario.days == n_days
    assert len(all_scenario.g_demand) == n_days

    wb = xlwt.Workbook()
    capacity = wb.add_sheet("容量记录")
    for j, name in enumerate(["P_fc", "P_el", "P_pv", "P_hp", "P_bs", "H_hs", "P_eb", "G_ht"]):
        capacity.write(0, j, name)

    model = MultiTime_model(1100, 1 / 6, save_xls=False, dump_debug=False)
    rows = []
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    model.formulate_MP(all_scenario.main_scenario)
    t_form = time.perf_counter() - t0
    log(f"formulate_MP {t_form:.2f}s  rss={peak_rss_mb():.0f} MB")

    stop_reason = None
    t0 = time.perf_counter()
    try:
        model.solve_MP(wb, capacity, 0)
    except RuntimeError as exc:
        log(f"solve_MP iter0 failed: {exc}")
        summary = {
            "n_days": n_days,
            "smoke": smoke,
            "max_iter": max_iter,
            "n_rounds": 0,
            "stop_reason": str(exc),
            "final_obj": None,
            "final_infeasible": None,
            "total_s": time.perf_counter() - wall0,
            "peak_rss_mb": peak_rss_mb(),
            "timing_csv": timing_path,
            "scenario_days_json": days_path,
            "capacities": getattr(model, "planning_res", None),
        }
        with open(f"res/pho_summary_{tag}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"done total={summary['total_s']:.1f}s peak_rss={summary['peak_rss_mb']:.0f}MB -> {timing_path}")
        return
    t_mp = time.perf_counter() - t0
    obj = float(model.MP.objVal)
    log(f"solve_MP iter0 (EBO) obj={obj:.2f}  {t_mp:.2f}s  rss={peak_rss_mb():.0f} MB")
    log(f"capacities {model.planning_res}")

    round_idx = 0
    fail_case = n_days
    while True:
        fail_case = 0
        t_sim = 0.0
        t_sp = 0.0
        sim_excess = 0.0
        sp_objs = []
        operation_revise = {"fail": False}
        failed_days = []

        for i in range(all_scenario.days):
            t1 = time.perf_counter()
            operation_revise = model.simulation_heat(
                model.operation_res[i],
                model.operation_res_5min[i],
                all_scenario.sub_scenario[i],
                operation_revise,
                i + 1,
                round_idx,
            )
            t_sim += time.perf_counter() - t1
            if not operation_revise["fail"]:
                continue
            fail_case += 1
            failed_days.append(i)
            sim_excess += sim_excess_kwh(operation_revise)
            if smoke:
                continue
            t2 = time.perf_counter()
            try:
                sp_obj = model.add_cut(
                    model.operation_res[i],
                    all_scenario.sub_scenario[i],
                    operation_revise,
                    i,
                )
            except RuntimeError as exc:
                log(f"add_cut day {i} failed: {exc}")
                sp_obj = None
            t_sp += time.perf_counter() - t2
            sp_objs.append(float(sp_obj) if sp_obj is not None else 0.0)

        row = {
            "round": round_idx,
            "stage": "EBO" if round_idx == 0 else "PHO",
            "obj": obj,
            "n_infeasible": fail_case,
            "sim_excess_kwh": sim_excess,
            "sp_obj_sum": float(sum(sp_objs)) if sp_objs else 0.0,
            "t_form_s": t_form if round_idx == 0 else 0.0,
            "t_mp_s": t_mp,
            "t_sim_s": t_sim,
            "t_sp_s": t_sp,
            "t_round_s": t_mp + t_sim + t_sp,
            "t_total_s": time.perf_counter() - wall0,
            "rss_mb": peak_rss_mb(),
            "failed_days": ";".join(str(d) for d in failed_days),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(timing_path, index=False)
        log(
            f"round {round_idx} {row['stage']} infeas={fail_case} "
            f"excess={sim_excess:.2f} sp={row['sp_obj_sum']:.2f} "
            f"sim={t_sim:.1f}s sp={t_sp:.2f}s rss={row['rss_mb']:.0f}MB"
        )

        if smoke:
            log("smoke: stop after EBO verification")
            break
        if fail_case == 0:
            log("all scenarios feasible")
            break
        if round_idx >= max_iter:
            log(f"hit max_iter={max_iter}")
            break

        round_idx += 1
        t0 = time.perf_counter()
        try:
            model.solve_MP(wb, capacity, round_idx)
        except RuntimeError as exc:
            stop_reason = str(exc)
            log(f"solve_MP iter{round_idx} failed: {exc}")
            break
        t_mp = time.perf_counter() - t0
        obj = float(model.MP.objVal)
        log(f"solve_MP iter{round_idx} obj={obj:.2f}  {t_mp:.2f}s  rss={peak_rss_mb():.0f} MB")

    summary = {
        "n_days": n_days,
        "smoke": smoke,
        "max_iter": max_iter,
        "n_rounds": len(rows),
        "stop_reason": stop_reason,
        "final_obj": rows[-1]["obj"] if rows else None,
        "final_infeasible": rows[-1]["n_infeasible"] if rows else None,
        "total_s": time.perf_counter() - wall0,
        "peak_rss_mb": max(r["rss_mb"] for r in rows) if rows else peak_rss_mb(),
        "timing_csv": timing_path,
        "scenario_days_json": days_path,
        "capacities": getattr(model, "planning_res", None),
    }
    with open(f"res/pho_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"done total={summary['total_s']:.1f}s peak_rss={summary['peak_rss_mb']:.0f}MB -> {timing_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = p.parse_args()
    run(args.days, args.max_iter, args.smoke, args.seed)


if __name__ == "__main__":
    main()
