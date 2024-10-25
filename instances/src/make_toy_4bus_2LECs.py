import os, random
import pandas as pd
import numpy as np

# Import power system library
import shilps.powersystems as ps  # https://github.com/jsepulvea/shilps
# Import rb_dnep objects
from rb_dnep import DataLEC, ProblemData, TimeConfig, DataTimeSeries
from rb_dnep.instances.time_series_generation import generate_demand_df, generate_pv_power_df


def build_toy_4bus_2LECs():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(THIS_DIR, "../data/toy_4bus_2LECs")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    #******************************************************************************#
    # Instatiate a ProblemData object
    #******************************************************************************#
    pdata = ProblemData()

    #******************************************************************************#
    # Load standard grid data
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
                        pmax_MW=ps.TSParameter(default=0.5))

    idx_new_dg1 = lec1.add(dg1)

    # Add non-existent generator for investment
    dginv1 = ps.InvGenerator(
        bus=0, snom_MVA= 0.5,
        pmax_MW=ps.TSParameter(default=0.5),
        cost_investment=1000.0,
        cost_maintenance=50.0
    )

    idx_new_dg2 = lec1.add(dginv1)

    # Create LEC 2
    lec2 = lec1.copy()
    lec2.name = "LEC 2"
    lec2.host_bus_idx = 2

    # Reinforcement capacity options
    lec1.reinforcement = [0.1, 0.2, 0.3]
    lec2.reinforcement = [0.1, 0.2, 0.3]

    # Add LECs to problem data
    pdata.add_lec(lec1, 1)
    pdata.add_lec(lec2, 2)

    # Set default time series names
    pdata.set_default_tsnames()

    #******************************************************************************#
    # Time series data
    #******************************************************************************#

    # Time Configuration
    # ------------------

    start = '2024-10-01'
    end = '2034-09-30'
    sampling_frequency = '15min'
    scenario_length = '1D'
    subperiod_starts = list(pd.date_range(start, end, freq='YE'))
    n_sce_per_subperiod = 2  # Scenarios per subperiod

    time_config = TimeConfig(
        start = start,
        end = end,
        sampling_frequency = sampling_frequency,
        scenario_length = scenario_length,
        subperiod_starts = subperiod_starts,
        n_sce_per_subperiod = n_sce_per_subperiod
    )
    # Build time series data
    # ----------------------
    scenarios = time_config.scenarios


    tsdata = DataTimeSeries(
        tsnames=pdata.tsnames(), time_config=time_config)

    # Generate demand timeseries host_grid
    l_demands = list(pdata.host_grid.demands.keys())
    l_demands_p_tsnames = [dem.p_MW.tsname for dem in pdata.host_grid.demands.values()]

    for scenario in scenarios:
        # Select random start within subperiod
        subperiod = time_config.scenario2subperiod(scenario)
        subperiod_start = time_config.subperiod_starts[subperiod]
        subperiod_end = time_config.subperiod_starts[subperiod] - pd.Timedelta(scenario_length)
        sce_start = random.sample(
            list(pd.date_range(subperiod_end, subperiod_start, freq=time_config.scenario_length)), 1)[0]
        
        sce_end = sce_start + time_config.scenario_length - time_config.sampling_frequency

        sampling_frequency = time_config.sampling_frequency
        
        df_d_hostgrid = generate_demand_df(
            sce_start, sce_end, sampling_frequency, l_demands_p_tsnames)

        # Scale demand
        scaling_factors = np.asarray([
            pdata.host_grid.demands[i].pnom_MW for i in l_demands])

        aux_max = np.max(df_d_hostgrid.values, axis=0)
        np.divide(df_d_hostgrid.values, aux_max, out=df_d_hostgrid.values)
        np.multiply(df_d_hostgrid.values, scaling_factors, out=df_d_hostgrid.values)

        tsdata[scenario].loc[:,l_demands_p_tsnames] = df_d_hostgrid.loc[:, l_demands_p_tsnames].values
        

    # Generate demand timeseries LECs

    for scenario in scenarios:
        # Select random start within subperiod
        subperiod = time_config.scenario2subperiod(scenario)
        subperiod_start = time_config.subperiod_starts[subperiod]
        subperiod_end = time_config.subperiod_starts[subperiod] - pd.Timedelta(scenario_length)
        sce_start = random.sample(
            list(pd.date_range(subperiod_end, subperiod_start, freq=time_config.scenario_length)), 1)[0]
        
        sce_end = sce_start + time_config.scenario_length - time_config.sampling_frequency

        sampling_frequency = time_config.sampling_frequency
        
        for lec_idx, lec in pdata.lecs.items():
            l_lec_demands_p_tsnames = [
                f"lec{lec_idx}_p_MW_{demand}" for demand in list(lec.demands.keys())]
            
            l_demands = list(lec.demands.keys())
            df_d_lec = generate_demand_df(
                sce_start, sce_end, sampling_frequency, l_lec_demands_p_tsnames)

            # Scale demand
            scaling_factors = np.asarray([
                lec.demands[i].pnom_MW for i in l_demands])

            aux_max = np.max(df_d_lec.values, axis=0)
            np.divide(df_d_lec.values, aux_max, out=df_d_lec.values)
            np.multiply(df_d_lec.values, scaling_factors, out=df_d_lec.values)

            tsdata[scenario].loc[:, l_lec_demands_p_tsnames] = df_d_lec.loc[:, l_lec_demands_p_tsnames].values
            

    # Generate PV timeseries LECs
    for scenario in scenarios:
        # Select random start within subperiod
        subperiod = time_config.scenario2subperiod(scenario)
        subperiod_start = time_config.subperiod_starts[subperiod]
        subperiod_end = time_config.subperiod_starts[subperiod] - pd.Timedelta(scenario_length)
        sce_start = random.sample(
            list(pd.date_range(subperiod_end, subperiod_start, freq=time_config.scenario_length)), 1)[0]
        
        sce_end = sce_start + time_config.scenario_length - time_config.sampling_frequency

        sampling_frequency = time_config.sampling_frequency

        for lec_idx, lec in pdata.lecs.items():
            l_lec_generators_p_tsnames = [f"lec{lec_idx}_pmax_MW_{generator}" for generator in list(lec.generators.keys())]
            l_lec_inv_generators_p_tsnames = [f"lec_inv_{lec_idx}_pmax_MW_{generator}" for generator in list(lec.generators.keys())]
            #l_lec_inv_generators_p_tsnames = [f""]
            l_generators = list(lec.generators.keys())

            df_pv_lec = generate_pv_power_df(sce_start, sce_end, sampling_frequency, l_lec_generators_p_tsnames)

            # Scale PV
            scaling_factors = np.asarray([
                lec.generators[i].snom_MVA for i in l_generators])

            aux_max = np.max(df_pv_lec.values, axis=0)
            np.divide(df_pv_lec.values, aux_max, out=df_pv_lec.values)
            np.multiply(df_pv_lec.values, scaling_factors, out=df_pv_lec.values)

            tsdata[scenario].loc[:, l_lec_generators_p_tsnames] = df_pv_lec.loc[:, l_lec_generators_p_tsnames].values
            
            tsdata[scenario].loc[:, l_lec_inv_generators_p_tsnames] = df_pv_lec.loc[:, l_lec_generators_p_tsnames].values

        


    pdata.tsdata = tsdata
    pdata.time_config = time_config

    # Write to files
    # --------------
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    prefix_name = os.path.join(OUTPUT_FOLDER, "toy_4bus_2LECs")
    tsdata_prefix = f"{prefix_name}_tsdata"
    pdata.write(prefix_name, tsdata_prefix)

    pdata.read(prefix_name + ".yaml", tsdata_prefix)

    return pdata
