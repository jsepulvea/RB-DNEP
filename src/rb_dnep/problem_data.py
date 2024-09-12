from dataclasses import dataclass, field
import shilps.powersystems as ps
import numpy as np
from enum import Enum
import pandas as pd
from typing import Dict, Tuple, List
import json
import os


class DataLEC(ps.DataPowerSystem):
    @classmethod
    def _initialize_class(cls):
        """This method is called when the class is first loaded, and it is used
        to register new serializable components and parameters that are
        specific to the subclass. The objective is for the to_dict and from_dict
        methods to work.

        To register new components containers, use the following syntax:
        ```python
        cls.register_serializable_component("new_component", NewComponent)
        ```

        To register new parameters, use the following syntax:
        ```python
        cls.register_serializable_parameter("new_param", int)
        ```

        """
        cls.register_serializable_parameter("host_bus_idx", int)
        cls.register_serializable_parameter("reinforcement", List[float])
        

    def __init__(self, host_bus_idx:int, index:int = None,
                 reinforcement: Dict[int, List[float]] = None,
                 **kwargs) -> None:
        """DataLEC is a subclass of DataPowerSystem that represents a Local
        Energy Community (LEC).  It is used to represent the LEC in the RB-DNEP
        problem formulation.

        Parameters
        ----------
        host_bus_idx : int
            The index of the bus in the host grid where the LEC is connected.
        index : int, optional
            Unique identifier for the LEC.
        reinforcement : Dict[int, List[float]], optional
            A dictionary that maps the index of the LEC to a list of reinforcement
            options.
        **kwargs
            Additional keyword arguments passed to the parent class.

        Notes
        -----
        The rest of the parameters are the same as DataPowerSystem.
        """
        self.host_bus_idx = host_bus_idx
        self.reinforcement = reinforcement if reinforcement is not None else {}

        super().__init__(**kwargs)

class ProblemData:
    def __init__(self, lecs: Dict[int, DataLEC]=None, host_grid: ps.DataPowerSystem = None,
                 tsdata: Dict[int, ps.DataTimeSeries] = None, time_config: ps.TimeConfig = None) -> None:
        """ProblemData is a class that represents the data of the RB-DNEP problem.

        Parameters
        ----------
        lecs : Dict[int, DataLEC], optional
            A dictionary that maps the index of the LEC to the DataLEC object.
        host_grid : ps.DataPowerSystem, optional
            The host grid of the problem.
        tsdata : Dict[int, ps.DataTimeSeries], optional
            A dictionary that maps the index of the LEC to the DataTimeSeries object.
        time_config : ps.TimeConfig, optional
            The time configuration of the problem.

        """

        self.lecs = lecs if lecs is not None else {}
        self.host_grid = host_grid
        self.tsdata = tsdata if tsdata is not None else {}
        self.time_config = time_config

    def system_dict(self):
        json_data = {}

        # Write host grid
        json_data["host_grid"] = self.host_grid.to_dict()

        # Write LECs
        json_data["lecs"] = {idx: lec.to_dict() for idx, lec in self.lecs.items()}

        if self.time_config is not None:
            json_data["time_config"] = self.time_config.to_dict()

        return json_data


    def write(self, json_path: str, tsdata_path: str) -> None:
        """Write the problem data to a file.

        Parameters:
        -----------
        - json_path: str, the path to the json file.
        - tsdata_path: str, the path to the time series data file.
        """
        json_data = {}

        # Write host grid
        json_data["host_grid"] = self.host_grid.to_dict()

        # Write LECs
        json_data["lecs"] = {idx: lec.to_dict() for idx, lec in self.lecs.items()}

        if self.time_config is not None:
            json_data["time_config"] = self.time_config.to_dict()

        json.dump(json_data, open(json_path, "w"))

        # Write time series data
        if self.tsdata is not None:
            self.tsdata.write(tsdata_path)

    @classmethod
    def read(cls, json_path: str, tsdata_path: str) -> 'ProblemData':
        """Read the problem data from a file.

        Parameters:
        -----------
        - json_path: str, the path to the json file.
        - tsdata_path: str, the path to the time series data file.
        """
        json_data = json.load(open(json_path, "r"))

        # Read host grid
        host_grid = ps.DataPowerSystem.from_dict(json_data["host_grid"])

        # Read LECs
        lecs = {int(idx): DataLEC.from_dict(lec)
                for idx, lec in json_data["lecs"].items()}

        # Read time configuration
        time_config = ps.TimeConfig.from_dict(json_data["time_config"])

        # Read time series data
        tsdata = ps.DataTimeSeries.read(tsdata_path)

        return cls(lecs=lecs, host_grid=host_grid, tsdata=tsdata,
                   time_config=time_config)


    def tsnames(self) -> List[str]:
        """Return the time series names of the problem data.

        Returns:
        --------
        - List[str], the time series names.
        """
        tsnames = []

        # Host grid
        for demand in self.host_grid.demands.values():
            tsnames.append(demand.p_MW.tsname)
            tsnames.append(demand.q_MVAr.tsname)

        for generator in self.host_grid.generators.values():
            tsnames.append(generator.pmax_MW.tsname)

        # LECs
        for lec in self.lecs.values():
            for demand in lec.demands.values():
                tsnames.append(demand.p_MW.tsname)
                tsnames.append(demand.q_MVAr.tsname)

            for generator in lec.generators.values():
                tsnames.append(generator.pmax_MW.tsname)

        return tsnames
