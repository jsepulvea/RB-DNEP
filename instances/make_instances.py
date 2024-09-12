import os
from src.toy_4bus_2LECs import build_toy_4bus_2LECs

DATA_DIR = "data"

pdata = build_toy_4bus_2LECs(os.path.join(DATA_DIR, "toy_4bus_2LECs"), start_date = '2023-01-01',
                         end_date = '2023-01-07', frequency = '15min') # frequency = '1h' for hour

import pandas as pd
import matplotlib.pyplot as plt


df = pdata.tsdata[1]

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
