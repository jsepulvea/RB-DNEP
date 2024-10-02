import os
from rb_dnep import ProblemData
import matplotlib.pyplot as plt


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

# Print current problem data
# ==========================
print(pdata.host_grid)
print(pdata.lecs)

df = pdata.tsdata[1]  # Get time series data for scenario 1

# Plotting the DataFrame
plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#******************************************************************************#
# Build and solve model
#******************************************************************************#



#******************************************************************************#
# Post process
#******************************************************************************#
