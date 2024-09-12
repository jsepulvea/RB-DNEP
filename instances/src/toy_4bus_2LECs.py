from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import shilps.powersystems as ps
import os

from rb_dnep.instances.time_series_generation import generate_demand_df, generate_pv_power_df
from rb_dnep.problem_data import DataLEC, ProblemData


def build_toy_4bus_2LECs(output_folder: str, start_date = '2023-01-01',
                         end_date = '2023-01-07', frequency = '1h'):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #******************************************************************************#
    # Time config
    #******************************************************************************#
    pdata = ProblemData()

    #******************************************************************************#
    # Distribution system data
    #******************************************************************************#
    dsdata = ps.case_ieee4bus()  # Distribution system data

    # Create new branch to be expanded
    dsdata.add(ps.InvBranch(busf=1, bust=2, r_pu=0.0162, x_pu=0.006,
        b_pu=0.0, snom_MVA=999.0, cost_investment = 1000.0, cost_maintenance = 0.0))

    pdata.host_grid = dsdata

    #******************************************************************************#
    # LECs data
    #******************************************************************************#

    # Create LEC 1
    lec1 = DataLEC(host_bus_idx=1, name="LEC 1")

    # Add demand
    lec1_demand = dsdata.demands[2].copy()  # Copy
    lec1_demand.index = None

    lec1.add(lec1_demand.scale(0.2, inplace=True))

    # Add existing generator
    dg1 = ps.PVGenerator(bus=0, snom_MVA= 0.5,
                        pmax_MW=ps.TSParameter("ds_p_MW1", 0.5))

    idx_new_dg1 = lec1.add(dg1)

    # Add new generator
    dginv1 = ps.InvGenerator(
        bus=0, snom_MVA= 0.5,
        pmax_MW=ps.TSParameter("ds_p_MW1", 0.5),
        cost_investment=1000.0,
        cost_maintenance=50.0
    )

    idx_new_dg2 = lec1.add(dginv1)

    lec2 = lec1.copy()
    lec2.name = "LEC 2"
    lec2.host_bus_idx = 2

    # Add LECs to problem data
    pdata.lecs = {1: lec1, 2: lec2}

    # Reinforcement capacity options
    lec1.reinforcement = [0.1, 0.2, 0.3]
    lec2.reinforcement = [0.1, 0.2, 0.3]

    #******************************************************************************#
    # Time series data
    #******************************************************************************#
    # Define time series names
    l_demands_p_tsnames = []
    for demand in dsdata.demands.values():
        p_tsname = f"ds_p_MW{demand.index}"
        q_tsname = f"ds_q_MVAr{demand.index}"
        demand.p_MW.set_tsname(p_tsname)
        l_demands_p_tsnames.append(p_tsname)
        demand.q_MVAr.set_tsname(q_tsname)

    for generator in dsdata.generators.values():
        generator.pmax_MW.set_tsname(f"ds_p_MW{generator.index}")

    for lec_idx, lec in pdata.lecs.items():
        for demand in lec.demands.values():
            demand.p_MW.set_tsname(f"lec{lec_idx}_p_MW{demand.index}")
            demand.q_MVAr.set_tsname(f"lec{lec_idx}_q_MVAr{demand.index}")

        for generator in lec.generators.values():
            generator.pmax_MW.set_tsname(f"lec{lec_idx}_p_MW{generator.index}")


    # Time config for raw timeseries generation


    time_config = ps.TimeConfig.from_dict({
        "start": start_date,
        "end": end_date,
        "freq": frequency
    })

    scenarios = list(range(1, 4))

    tsdata = ps.DataTimeSeries(
        tsnames=pdata.tsnames(), scenarios=scenarios, time_config=time_config)

    # Generate demand timeseries host_grid
    l_demands = list(dsdata.demands.keys())

    for scenario in scenarios:
        df_d_hostgrid = generate_demand_df(
            start_date, end_date, frequency, l_demands_p_tsnames)

        # Scale demand
        scaling_factors = np.asarray([
            dsdata.demands[i].pnom_MW for i in l_demands])

        aux_max = np.max(df_d_hostgrid.values, axis=0)
        np.divide(df_d_hostgrid.values, aux_max, out=df_d_hostgrid.values)
        np.multiply(df_d_hostgrid.values, scaling_factors, out=df_d_hostgrid.values)

        tsdata[scenario].loc[:,l_demands_p_tsnames] = df_d_hostgrid.loc[:, l_demands_p_tsnames].values

    # Generate demand timeseries LECs

    for scenario in scenarios:
        for lec_idx, lec in pdata.lecs.items():
            l_lec_demands_p_tsnames = [
                f"lec{lec_idx}_p_MW{demand}" for demand in list(lec.demands.keys())]
            
            l_demands = list(lec.demands.keys())
            df_d_lec = generate_demand_df(
                start_date, end_date, frequency, l_lec_demands_p_tsnames)

            # Scale demand
            scaling_factors = np.asarray([
                lec.demands[i].pnom_MW for i in l_demands])

            aux_max = np.max(df_d_lec.values, axis=0)
            np.divide(df_d_lec.values, aux_max, out=df_d_lec.values)
            np.multiply(df_d_lec.values, scaling_factors, out=df_d_lec.values)

            tsdata[scenario].loc[:, l_lec_demands_p_tsnames] = df_d_lec.loc[:, l_lec_demands_p_tsnames].values


    # Generate PV timeseries LECs
    for scenario in scenarios:
        for lec_idx, lec in pdata.lecs.items():
            l_lec_generators_p_tsnames = [f"lec{lec_idx}_p_MW{generator}" for generator in list(lec.generators.keys())]
            l_generators = list(lec.generators.keys())
            n_generators = len(l_generators)

            df_pv_lec = generate_pv_power_df(start_date, end_date, frequency, l_lec_generators_p_tsnames)

            # Scale PV
            scaling_factors = np.asarray([
                lec.generators[i].snom_MVA for i in l_generators])

            aux_max = np.max(df_pv_lec.values, axis=0)
            np.divide(df_pv_lec.values, aux_max, out=df_pv_lec.values)
            np.multiply(df_pv_lec.values, scaling_factors, out=df_pv_lec.values)

            tsdata[scenario].loc[:, l_lec_generators_p_tsnames] = df_pv_lec.loc[:, l_lec_generators_p_tsnames].values



    pdata.tsdata = tsdata
    pdata.time_config = time_config

    prefix_name = os.path.join(output_folder, "toy_4bus_2LECs")
    system_data_path = f"{prefix_name}.json"
    tsdata_prefix = f"{prefix_name}_tsdata"
    pdata.write(system_data_path, tsdata_prefix)

    return pdata
