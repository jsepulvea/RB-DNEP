# Import "standard" libraries
import os, random
import pandas as pd
import numpy as np

# Import rb_dnep objects
from rb_dnep import DataLEC, ProblemData, TimeConfig, DataTimeSeries
from rb_dnep.instances.time_series_generation import generate_demand_df, generate_pv_power_df


#******************************************************************************#
# Load grid template
#******************************************************************************#
# Set input-output files
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_TEMPLATE_FILE = os.path.join(
    THIS_DIR,
    "example_data/toy_4bus_2LECs_template.json"
)
OUTPUT_FOLDER = "example_data/toy_4bus_2LECs"

# Load grid template
pdata = ProblemData.read(GRID_TEMPLATE_FILE)

#******************************************************************************#
# Series data
#******************************************************************************#

# Time Config
# -----------

start = '2024-10-01'
end = '2034-09-30'
sampling_frequency = '15min'
scenario_length = '1D'

# Generate scenario_starts
n_scenarios_total = 25
all_days = pd.date_range(start, end, freq='D')
random_days = random.sample(list(all_days), min(n_scenarios_total, len(all_days)))
random_days = sorted(random_days)

time_config = TimeConfig(
    start = start,
    end = end,
    sampling_frequency = sampling_frequency,
    scenario_length = scenario_length,
    scenario_starts = random_days
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

    sce_start = time_config.scenario_starts[scenario]
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
    tsdata[scenario].index = pd.date_range(sce_start, sce_end, freq=time_config.sampling_frequency)
    

# Generate demand timeseries LECs

for scenario in scenarios:
    sce_start = time_config.scenario_starts[scenario]
    sce_end = sce_start + time_config.scenario_length - time_config.sampling_frequency
    sampling_frequency = time_config.sampling_frequency
    
    for lec_idx, lec in pdata.lecs.items():
        l_lec_demands_p_tsnames = [
            f"lec{lec_idx}_p_MW{demand}" for demand in list(lec.demands.keys())]
        
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
        tsdata[scenario].index = pd.date_range(sce_start, sce_end, freq=time_config.sampling_frequency)

# Generate PV timeseries LECs
for scenario in scenarios:
    sce_start = time_config.scenario_starts[scenario]
    sce_end = sce_start + time_config.scenario_length - time_config.sampling_frequency
    sampling_frequency = time_config.sampling_frequency

    for lec_idx, lec in pdata.lecs.items():
        l_lec_generators_p_tsnames = [f"lec{lec_idx}_p_MW{generator}" for generator in list(lec.generators.keys())]
        l_generators = list(lec.generators.keys())

        df_pv_lec = generate_pv_power_df(sce_start, sce_end, sampling_frequency, l_lec_generators_p_tsnames)

        # Scale PV
        scaling_factors = np.asarray([
            lec.generators[i].snom_MVA for i in l_generators])

        aux_max = np.max(df_pv_lec.values, axis=0)
        np.divide(df_pv_lec.values, aux_max, out=df_pv_lec.values)
        np.multiply(df_pv_lec.values, scaling_factors, out=df_pv_lec.values)

        tsdata[scenario].loc[:, l_lec_generators_p_tsnames] = df_pv_lec.loc[:, l_lec_generators_p_tsnames].values
        tsdata[scenario].index = pd.date_range(sce_start, sce_end, freq=time_config.sampling_frequency)


pdata.tsdata = tsdata
pdata.time_config = time_config

# Write to files
# --------------
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

prefix_name = os.path.join(OUTPUT_FOLDER, "toy_4bus_2LECs")
system_data_path = f"{prefix_name}.json"
tsdata_prefix = f"{prefix_name}_tsdata"

pdata.write(system_data_path, tsdata_prefix)
