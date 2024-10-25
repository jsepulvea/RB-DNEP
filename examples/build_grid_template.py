import os
import shilps.powersystems as ps  # https://github.com/jsepulvea/shilps
from rb_dnep import DataLEC, ProblemData

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(THIS_DIR, "example_data")

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

# Save to file
prefix_name = os.path.join(OUTPUT_FOLDER, "toy_4bus_2LECs_template")
pdata_path = f"{prefix_name}.json"
pdata.write(pdata_path)
