=============================
RB-DNEP: Robust Bilevel DNEP
=============================

Overview
========
``RB-DNEP`` is a Python package designed to implement a robust bilevel optimization model for distribution network expansion planning. The package addresses the interdependent planning needs of active distribution networks (ADNs) and local energy communities (LECs), incorporating uncertainty in demand and solar irradiance fluctuations.

The model is structured into two levels: the upper-level problem, handled by the distribution system operator, involves investment decisions in infrastructure. The lower-level, tackled by local energy communities, focuses on the deployment of distributed energy resources, considering the constraints imposed by the upper-level decisions.

Overlead link:
https://www.overleaf.com/2772772453csqvbdyynbhb#1adfbe

Installation
============

To install this package, follow these steps:

.. note::
   This guide assumes that you have Python 3 installed on your system.

1. **Clone the repository:**

.. code-block:: bash

    git clone git@github.com:jsepulvea/RB-DNEP.git
    cd RB-DNEP 

2. **Create and activate a virtual environment (recommended):**

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate

3. **Install the package in editable mode using pip:**

.. code-block:: bash

    pip install -e .

This will install the package in editable mode, allowing you to make changes to the code and have them immediately reflected without reinstalling the package.

Usage
=====

``ProblemData`` Data Structure Documentation
============================================

The ``ProblemData`` class represents the data structure used in the
**Robust Bilevel Distribution Network Expansion Problem (RB-DNEP)**. It
encapsulates all necessary components to model and solve the RB-DNEP,
including the host grid, local energy communities (LECs), time series
data, and time configuration.

See the folder ``instances/data/toy_4bus_2LECs`` for a sample instance.
See examples ``examples/build_grid_instance.py`` and
``examples/build_scenarios.py`` for how to build a problem data instance.

Overview
--------

Components
~~~~~~~~~~

-  **Host Grid** (``host_grid``): An instance of ``DataPowerSystem``
   representing the main power system network, including buses, lines,
   generators, and demands.
-  **Local Energy Communities** (``lecs``): A dictionary mapping LEC
   indices to ``DataLEC`` instances. Each ``DataLEC`` represents a local
   energy community connected to the host grid.
-  **Time Series Data** (``tsdata``): An instance of ``DataTimeSeries``
   containing time-dependent data for demands and generation across
   different scenarios.
-  **Time Configuration** (``time_config``): An instance of
   ``TimeConfig`` defining temporal parameters like start and end times,
   sampling frequency, scenario length, and subperiod starts.

Class Structure
---------------

.. code:: python

   class ProblemData:
       def __init__(self, lecs=None, host_grid=None, tsdata=None, time_config=None):
           # Initialize components

Time Series Structure
---------------------

Host Grid Time Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Demand.p_MW**: Active power demands in MW at different nodes of the
   host grid.
-  **Demand.q_MVAr**: Reactive power demands in MVAr at different nodes
   of the host grid.
-  **Generator.pmax_MW**: Maximum power output capabilities of
   generators in MW.

LECs (Local Energy Communities) Time Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each LEC has its own set of time series data:

-  **Demand.p_MW**: Active power demands in MW at different nodes within
   the LEC.
-  **Demand.q_MVAr**: Reactive power demands in MVAr at different nodes
   within the LEC.
-  **Generator.pmax_MW**: Maximum power output capabilities of
   generators in MW within the LEC.

Required Time Series Names
~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance might require for example the following time series:

.. code:: yaml

   host_grid:
     Demand.p_MW:
       - ds_p_MW_0
       - ds_p_MW_1
       - ds_p_MW_2
     Demand.q_MVAr:
       - ds_q_MVAr_0
       - ds_q_MVAr_1
       - ds_q_MVAr_2
     Generator.pmax_MW: []
   lecs:
     1:
       Demand.p_MW:
         - lec1_p_MW_0
       Demand.q_MVAr:
         - lec1_q_MVAr_0
       Generator.pmax_MW:
         - lec1_pmax_MW_0
         - lec_inv_1_pmax_MW_0
     2:
       Demand.p_MW:
         - lec2_p_MW_0
       Demand.q_MVAr:
         - lec2_q_MVAr_0
       Generator.pmax_MW:
         - lec2_pmax_MW_0
         - lec_inv_2_pmax_MW_0

File Format
-----------

System Data Files
~~~~~~~~~~~~~~~~~

-  **Format**: YAML

-  **Contents**: Host grid data, LECs data, and time configuration.

-  **Writing**: Use the ``write`` method to serialize and save the data
   to a YAML file.

   .. code:: python

      problem_data.write(yaml_path='system_data.yaml')

-  **Reading**: Use the ``read`` class method to deserialize and load
   the data from a YAML file.

   .. code:: python

      problem_data = ProblemData.read(yaml_path='system_data.yaml')

Time Series Data Files
~~~~~~~~~~~~~~~~~~~~~~

-  **Format**: CSV

-  **Contents**: Time series data for demands and generation.

-  **File Naming Convention**: Each DataFrame corresponding to each scenario is
   saved as a separate CSV file named using a prefix and key identifier.

   ::

      tsdata_key{key_str}.csv

   Example:

   ::

      tsdata_key1.csv
      tsdata_key2.csv

-  **Writing**: Use the ``DataTimeSeries.write`` method to save time
   series data.

   .. code:: python

      problem_data.tsdata.write(prefix='tsdata')

-  **Reading**: Use the ``DataTimeSeries.read`` method to load time
   series data.

   .. code:: python

      tsdata = DataTimeSeries.read(prefix='tsdata')

