import os
from rb_dnep import ProblemData, DDNEPModel

#******************************************************************************#
# Load problem data
#******************************************************************************#

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FOLDER = os.path.join(THIS_DIR, "../instances/data/toy_4bus_2LECs")
INSTANCE_NAME = "toy_4bus_2LECs"

system_data_path = os.path.join(INSTANCE_FOLDER, INSTANCE_NAME + ".json")
timeseries_data_prefix = os.path.join(INSTANCE_FOLDER, INSTANCE_NAME + "_tsdata")

pdata = ProblemData.read(
    json_path=system_data_path, tsdata_path=timeseries_data_prefix)

#******************************************************************************#
# Build and solve model
#******************************************************************************#



#******************************************************************************#
# Post process
#******************************************************************************#
