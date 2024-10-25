from dataclasses import dataclass, field
import shilps.powersystems as ps
import numpy as np
from enum import Enum
import pandas as pd
from typing import Dict, Tuple, List, Any, Union
import json
import os
import glob
import yaml

import pandas as pd

#******************************************************************************#
# Utils
#******************************************************************************#

def convert_keys_to_int(data):
    """
    Convert all dictionary keys to integers if possible.
    """
    if isinstance(data, dict):
        return {
            int(key) if key.isdigit() else key: convert_keys_to_int(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_keys_to_int(item) for item in data]
    else:
        return data

def pretty_print_dict(d: dict) -> str:

    class NoAliasDumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True

    return yaml.dump(d, default_flow_style=False, sort_keys=False, Dumper=NoAliasDumper)

#******************************************************************************#
# Problem data classes
#******************************************************************************#
class TimeConfig:
    """
    A class representing the time configuration for a planning problem.

    Parameters
    ----------
    start : pd.Timestamp or str
        The start time of the planning horizon.
    end : pd.Timestamp or str
        The end time of the planning horizon.
    sampling_frequency : pd.Timedelta or str
        The sampling frequency for all scenarios.
    scenario_length : pd.Timedelta or str
        The length of each scenario.
    scenario_starts : List[Union[pd.Timestamp, str]]
        The starting timestamps of each scenario.
    """

    def __init__(
        self,
        start: Union[pd.Timestamp, str],
        end: Union[pd.Timestamp, str],
        sampling_frequency: Union[pd.Timedelta, str],
        scenario_length: Union[pd.Timedelta, str],
        scenario_starts: List[Union[pd.Timestamp, str]],
    ):
        """
        Initialize a TimeConfig instance.

        Parameters
        ----------
        start : pd.Timestamp or str
            The start time of the planning horizon.
        end : pd.Timestamp or str
            The end time of the planning horizon.
        sampling_frequency : pd.Timedelta or str
            The sampling frequency for all scenarios.
        scenario_length : pd.Timedelta or str
            The length of each scenario.
        scenario_starts : List[Union[pd.Timestamp, str]]
            The starting timestamps of each scenario.
        """
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.sampling_frequency = pd.Timedelta(sampling_frequency)
        self.scenario_length = pd.Timedelta(scenario_length)
        self.scenario_starts = [pd.Timestamp(ts) for ts in scenario_starts]
        self.scenario_indices = list(range(len(self.scenario_starts)))
    
    @property
    def scenarios(self):
        return self.scenario_indices
    
    def get_scenario_time_range(self, i: int) -> pd.DatetimeIndex:
        """
        Get the time range for the scenario at the specified index.

        Parameters
        ----------
        i : int
            The index of the scenario.

        Returns
        -------
        pd.DatetimeIndex
            The time range for the specified scenario.
        """
        if i < 0 or i >= len(self.scenario_starts):
            raise IndexError("Scenario index out of range.")
        
        start = self.scenario_starts[i]
        time_range = pd.date_range(
            start=start,
            periods=int(self.scenario_length / self.sampling_frequency),
            freq=self.sampling_frequency
        )
        return time_range

    def __str__(self):
        """
        Return a string representation of the TimeConfig instance.

        Returns
        -------
        str
            A string representation of the TimeConfig instance.
        """
        s = f"TimeConfig:\n"
        s += f"  Planning horizon start: {self.start}\n"
        s += f"  Planning horizon end: {self.end}\n"
        s += f"  Sampling frequency: {self.sampling_frequency}\n"
        s += f"  Scenario length: {self.scenario_length}\n"
        s += f"  Number of scenarios: {len(self.scenario_starts)}\n"
        s += f"  Scenarios:\n"
        for idx, start_time in zip(self.scenario_indices, self.scenario_starts):
            s += f"    Scenario {idx}: starts at {start_time}\n"
        return s

    def to_dict(self):
        """
        Convert the TimeConfig instance to a serializable dictionary.

        Returns
        -------
        dict
            A dictionary representation of the TimeConfig instance.
        """
        return {
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'sampling_frequency': str(self.sampling_frequency),
            'scenario_length': str(self.scenario_length),
            'scenario_starts': [ts.isoformat() for ts in self.scenario_starts],
            'scenario_indices': self.scenario_indices,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a TimeConfig instance from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the TimeConfig data.

        Returns
        -------
        TimeConfig
            A TimeConfig instance created from the dictionary data.
        """
        return cls(
            start=pd.Timestamp(data['start']),
            end=pd.Timestamp(data['end']),
            sampling_frequency=pd.Timedelta(data['sampling_frequency']),
            scenario_length=pd.Timedelta(data['scenario_length']),
            scenario_starts=[pd.Timestamp(ts) for ts in data['scenario_starts']],
        )

    def write(self, filename: str):
        """
        Serialize the TimeConfig instance to a JSON file.

        Parameters
        ----------
        filename : str
            The filename to write the JSON data to.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def read(cls, filename: str):
        """
        Deserialize a TimeConfig instance from a JSON file.

        Parameters
        ----------
        filename : str
            The filename to read the JSON data from.

        Returns
        -------
        TimeConfig
            A TimeConfig instance created from the JSON data.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    


class DataTimeSeries:
    def __init__(self, dict_df: Dict[Any, pd.DataFrame] = None, tsnames: str = None,
                 time_config:TimeConfig = None):
        
        self.tsnames = tsnames
        self.dict_df = dict_df if dict_df is not None else {}

        scenarios = time_config.scenarios

        for scenario in scenarios:
            
            time_range = time_config.get_scenario_time_range(scenario)

            self.dict_df[scenario] = self._create_empty_df(tsnames, time_range)

    @staticmethod
    def _create_empty_df(tsnames, time_range):
        return pd.DataFrame(index=time_range, columns=tsnames)
    
    def get_value(self, tsname, *args):
        scenario = args[0]
        key = args[1:]
        return self.dict_df[scenario].loc[key, tsname]

    def get_series(self, *args):
        """Retrieve a specific time series from the DataFrame corresponding to a
        key.

        Parameters
        ----------
        *args : tuple
            A variable-length argument list where all elements except the last
            one are used to form the key (as a tuple) for accessing the
            dictionary. The last element of `args` is the name of the time
            series to retrieve from the DataFrame.

        Returns
        -------
        pandas.Series
            The requested time series from the DataFrame associated with the
            specified key.

        Raises
        ------
        KeyError
            If the specified key does not exist in `self.dict_df`.
        """
        dict_key = tuple(args[:-1])
        tsname = args[-1]
        return self.dict_df[dict_key].loc[:, tsname]

    def set_series(self, *args, value):
        """Set a specific time series in the DataFrame corresponding to a key.

        Parameters
        ----------
        *args : tuple
            A variable-length argument list where all elements except the last
            one are used to form the key (as a tuple) for accessing the
            dictionary. The last element of `args` is the name of the time
            series to set in the DataFrame.

        value : array-like
            The values to assign to the specified time series in the DataFrame.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the specified key does not exist in `self.dict_df`.
        ValueError
            If the length of `value` does not match the length of the
            DataFrame's index.
        """
        dict_key = tuple(args[:-1])
        tsname = args[-1]
        self.dict_df[dict_key].loc[:, tsname] = value

    def keys(self):
        return self.dict_df.keys()

    def __getitem__(self, key):
        return self.dict_df[key]
    
    def __setitem__(self, key, value):
        self.dict_df[key] = value

    def __str__(self) -> str:
        result = []
        for key, df in self.dict_df.items():
            result.append(f"Key: {key}")
            result.append(str(df))
        return "\n".join(result)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _key2str(key):
        if isinstance(key, int):
            return str(key)
        return "_".join(map(str, key))

    @staticmethod
    def _str2key(key_str):
        if key_str.isdigit():
            return int(key_str)
        return tuple(map(int, key_str.split("_")))

    def write(self, prefix: str):
        """Write the DataFrames in `self.dict_df` to CSV files.

        Each DataFrame in `self.dict_df` is saved to a separate CSV file. The
        files are named using the provided `prefix` followed by the string
        representation of the key used in the dictionary.

        The key string pattern converts a tuple of integers (used as keys in the
        dictionary) into a string by joining the integers with underscores. For
        example, the tuple `(1, 2, 3)` becomes the string `"1_2_3"`. This string
        is then used in filenames to uniquely identify and relate the files to
        their corresponding dictionary keys.

        Parameters
        ----------
        prefix : str
            The prefix to use for naming the output files. Each file will be
            named as `prefix_key{key_str}.csv`, where `key_str` is the string
            representation of the dictionary key.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If there is an issue writing the files to the disk.
        """
        for key, df in self.dict_df.items():
            key_str = self._key2str(key)
            df.to_csv(f"{prefix}_key{key_str}.csv", index_label="time")

    @classmethod
    def read(cls, prefix: str, format_ext="csv"):
        """Read DataFrames from files with the specified prefix and format, and
        reconstruct a `DataTimeSeries` object.

        This method searches for files with the given `prefix` and file
        extension (`format_ext`), reads them, and reconstructs the
        `DataTimeSeries` object with a dictionary where keys are derived from
        the filenames.

        The key string pattern converts a string back into the original tuple of
        integers by splitting the string on underscores. For example, the string
        `"1_2_3"` is converted back into the tuple `(1, 2, 3)`.

        Parameters
        ----------
        prefix : str
            The prefix used to identify the files to be read. The method expects
            files to follow the naming convention
            `prefix_key{key_str}.{format_ext}`.

        format_ext : str, optional
            The file format extension to look for (default is "csv").

        Returns
        -------
        DataTimeSeries
            A new instance of `DataTimeSeries` with the dictionary of DataFrames
            reconstructed from the files.

        Raises
        ------
        ValueError
            If no files with the specified prefix and format are found.

        IOError
            If there is an issue reading the files.
        """
        dict_df = {}
        file_pattern = f"{prefix}_key*.{format_ext}"
        files = glob.glob(file_pattern)

        if not files:
            raise ValueError(f"No files with prefix {prefix} found.")

        for file in files:
            # Extract key part from filename
            file_basename = os.path.basename(file)
            file_basename_noext = file_basename.rsplit("." + format_ext)[0]    
            key_str = file_basename_noext.rsplit("key")[1]

            key = cls._str2key(key_str)
            dict_df[key] = pd.read_csv(file, index_col="time")
            dict_df[key].index = pd.to_datetime(dict_df[key].index)

        return cls(dict_df)

    @staticmethod
    def check_files_with_fullpath_prefix(fullpath_prefix):
        """Checks if there are files corresponding to the given full path
        prefix."""
        matching_files = glob.glob(fullpath_prefix + "*")
        return bool(matching_files)



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
    """
    
    """

    def __init__(self, lecs: Dict[int, DataLEC]=None, host_grid: ps.DataPowerSystem = None,
                 tsdata: DataTimeSeries = None, time_config: ps.PlanningTimeConfig = None) -> None:
        """ProblemData is a class that represents the data of the RB-DNEP problem.

        Parameters
        ----------
        lecs : Dict[int, DataLEC], optional
            A dictionary that maps the index of the LEC to the DataLEC object.
        host_grid : ps.DataPowerSystem, optional
            The host grid of the problem.
        tsdata : Dict[int, ps.DataTimeSeries], optional
            A dictionary that maps the index of the LEC to the DataTimeSeries object.
        time_config : ps.PlanningTimeConfig, optional
            The time configuration of the problem.

        """

        self.lecs = lecs if lecs is not None else {}
        self.host_grid = host_grid
        self.tsdata = tsdata if tsdata is not None else {}
        self.time_config = time_config

    def add_lec(self, lec: DataLEC, index: int) -> None:
        """Add a LEC to the problem data.

        Parameters
        ----------
        lec : DataLEC
            The LEC to add to the problem data.
        index : int
            The index of the LEC.
        """
        self.lecs[index] = lec

    def system_dict(self):
        json_data = {}

        # Write host grid
        json_data["host_grid"] = self.host_grid.to_dict()

        # Write LECs
        json_data["lecs"] = {idx: lec.to_dict() for idx, lec in self.lecs.items()}

        if self.time_config is not None:
            json_data["time_config"] = self.time_config.to_dict()

        return json_data    

    def __str__(self):
        pretty_str = "Problem data:\n-------------\n\n"
        pretty_str += json.dumps(self.system_dict(), indent=4)
        pretty_str += "\nTime series data:\n-----------------\n\n"
        pretty_str += str(self.tsdata)
        return pretty_str
    
    def __repr__(self):
        return self.__str__()

    def write(self, json_path: str, tsdata_path: str=None) -> None:
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
        if self.tsdata:
            self.tsdata.write(tsdata_path)

    @classmethod
    def read(cls, json_path: str, tsdata_path: str=None) -> 'ProblemData':
        """Read the problem data from a file.

        Parameters:
        -----------
        - json_path: str, the path to the json file.
        - tsdata_path: str, the path to the time series data file.
        """
        json_data = json.load(open(json_path, "r"))
        json_data = convert_keys_to_int(json_data)

        # Read host grid
        host_grid = ps.DataPowerSystem.from_dict(json_data["host_grid"])

        # Read LECs
        lecs = {int(idx): DataLEC.from_dict(lec)
                for idx, lec in json_data["lecs"].items()}

        # Read time configuration
        if "time_config" in json_data:
            time_config = ps.PlanningTimeConfig.from_dict(json_data["time_config"])
        else:
            time_config = None

        # Read time series data
        if tsdata_path is not None:
            tsdata = ps.DataTimeSeries.read(tsdata_path)
        else:
            tsdata = None

        return cls(lecs=lecs, host_grid=host_grid, tsdata=tsdata,
                   time_config=time_config)

    def set_default_tsnames(self):
        
        l_demands_p_tsnames = []
        for demand in self.host_grid.demands.values():
            p_tsname = f"ds_p_MW{demand.index}"
            q_tsname = f"ds_q_MVAr{demand.index}"
            demand.p_MW.set_tsname(p_tsname)
            l_demands_p_tsnames.append(p_tsname)
            demand.q_MVAr.set_tsname(q_tsname)

        for generator in self.host_grid.generators.values():
            generator.pmax_MW.set_tsname(f"ds_p_MW{generator.index}")

        for lec_idx, lec in self.lecs.items():
            for demand in lec.demands.values():
                demand.p_MW.set_tsname(f"lec{lec_idx}_p_MW{demand.index}")
                demand.q_MVAr.set_tsname(f"lec{lec_idx}_q_MVAr{demand.index}")

            for generator in lec.generators.values():
                generator.pmax_MW.set_tsname(f"lec{lec_idx}_pmax_MW{generator.index}")

            for invgenerator in lec.inv_generators.values():
                invgenerator.pmax_MW.set_tsname(f"lec_inv_{lec_idx}_pmax_MW{generator.index}")


    def time_series_structure(self) -> List[str]:

        aux_tstypes = {"Demand.p_MW": [], "Demand.q_MVAr": [], "Generator.pmax_MW": []}

        ts_structure = {
            "host_grid": aux_tstypes.copy(),
            "lecs": {lec_index: aux_tstypes.copy() for lec_index in self.lecs.keys()}
        }

        # Host grid
        for demand in self.host_grid.demands.values():
            ts_structure["host_grid"]["Demand.p_MW"].append(demand.p_MW.tsname)
            ts_structure["host_grid"]["Demand.q_MVAr"].append(demand.q_MVAr.tsname)

        for generator in self.host_grid.generators.values():
            ts_structure["host_grid"]["Generator.pmax_MW"].append(generator.pmax_MW.tsname)

        # LECs
        for lec_idx, lec in self.lecs.items():
            for demand in lec.demands.values():
                ts_structure["lecs"][lec_idx]["Demand.p_MW"].append(demand.p_MW.tsname)
                ts_structure["lecs"][lec_idx]["Demand.q_MVAr"].append(demand.q_MVAr.tsname)

            for generator in lec.generators.values():
                ts_structure["lecs"][lec_idx]["Generator.pmax_MW"].append(generator.pmax_MW.tsname)

        return ts_structure
    
    def display_time_series_structure(self):
        ts_structure = self.time_series_structure()
        pretty_str = "Time series structure:\n----------------------\n\n"
        print(pretty_str + pretty_print_dict(ts_structure))
    
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

            for invgenerator in lec.inv_generators.values():
                tsnames.append(invgenerator.pmax_MW.tsname)

        return tsnames
    
    def validate(self) -> bool:
        """
        Validate the ProblemData instance.

        - Checks if there are duplicate time series names in the list returned
        by the tsnames() method. If duplicates are found, the instance is
        considered invalid.

        Returns
        -------
        bool
            True if the ProblemData instance is valid (no duplicate tsnames),
            False otherwise.
        """
        tsnames_list = self.tsnames()
        unique_tsnames = set(tsnames_list)
        if len(tsnames_list) != len(unique_tsnames):
            print("Validation failed: Duplicate time series names found.")
            return False
        return True
