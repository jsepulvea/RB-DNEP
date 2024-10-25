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

# Define custom YAML tag for pandas.Timestamp
TIMESTAMP_TAG = '!timestamp'

# Custom representer for pandas.Timestamp
def timestamp_representer(dumper, data):
    return dumper.represent_scalar(TIMESTAMP_TAG, data.isoformat())

# Custom constructor for pandas.Timestamp
def timestamp_constructor(loader, node):
    value = loader.construct_scalar(node)
    return pd.Timestamp(value)

# Register the custom representer and constructor with PyYAML
yaml.add_representer(pd.Timestamp, timestamp_representer)
yaml.add_constructor(TIMESTAMP_TAG, timestamp_constructor)

# Define custom YAML tag for pandas.Timedelta
TIMEDELTA_TAG = '!timedelta'

# Custom representer for pandas.Timedelta
def timedelta_representer(dumper, data):
    # Convert the Timedelta to ISO 8601 duration string format
    return dumper.represent_scalar(TIMEDELTA_TAG, str(data))

# Custom constructor for pandas.Timedelta
def timedelta_constructor(loader, node):
    value = loader.construct_scalar(node)
    return pd.Timedelta(value)

# Register the custom representer and constructor with PyYAML
yaml.add_representer(pd.Timedelta, timedelta_representer)
yaml.add_constructor(TIMEDELTA_TAG, timedelta_constructor)


#******************************************************************************#
# Problem data classes
#******************************************************************************#
@dataclass
class TimeConfig(ps.SerializableDataClass):
    """
    A class representing the time configuration for a planning problem.
    """
    start: Union[pd.Timestamp, str]
    end: Union[pd.Timestamp, str]
    sampling_frequency: Union[pd.Timedelta, str]
    scenario_length: Union[pd.Timedelta, str]
    subperiod_starts: Union[pd.Timedelta, str]
    n_sce_per_subperiod: int

    def __post_init__(self):
        # Convert string inputs to the appropriate pandas time objects
        self.start = pd.Timestamp(self.start) if isinstance(self.start, str) else self.start
        self.end = pd.Timestamp(self.end) if isinstance(self.end, str) else self.end
        self.sampling_frequency = pd.to_timedelta(self.sampling_frequency) if isinstance(self.sampling_frequency, str) else self.sampling_frequency
        self.scenario_length = pd.to_timedelta(self.scenario_length) if isinstance(self.scenario_length, str) else self.scenario_length
        self.subperiod_starts = pd.to_timedelta(self.subperiod_starts) if isinstance(self.subperiod_starts, str) else self.subperiod_starts

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
        s += f"  Subperiod starts:\n"
        for i, subperiod_start in enumerate(self.subperiod_starts):
            s += f"    Subperiod {i}: {subperiod_start}\n"
        s += f"  Number of scenarios per subperiod: {self.n_sce_per_subperiod}\n"
        return s

    def __repr__(self):
        """
        Return a string representation of the TimeConfig instance.

        Returns
        -------
        str
            A string representation of the TimeConfig instance.
        """
        return self.__str__()

    @property
    def n_subperiods(self) -> int:
        return len(self.subperiod_starts)

    @property
    def n_scenarios(self) -> int:
        return self.n_sce_per_subperiod * self.n_subperiods
    
    @property
    def scenarios(self) -> List[int]:
        return list(range(self.n_scenarios))
    
    @property
    def scenario_dimension(self) -> int:
        return self.scenario_length // self.sampling_frequency
    
    def scenario2subperiod(self, sce_idx) -> int:
        return sce_idx // self.n_sce_per_subperiod

    def is_valid(self) -> bool:
        """
        Validate if the configuration is logically consistent:
        - The start time is before the end time.
        - The scenario length is less than the planning horizon length.
        - The subperiod length is a multiple of the scenario length.

        Returns
        -------
        bool
            True if the configuration is valid, otherwise False.
        """
        start_time = pd.Timestamp(self.start)
        end_time = pd.Timestamp(self.end)
        if start_time >= end_time:
            return False
        if pd.Timedelta(self.scenario_length) > (end_time - start_time):
            return False
        
        # Check if the scenario length is a multiple of the sampling frequency
        if pd.Timedelta(self.scenario_length) % pd.Timedelta(self.sampling_frequency) != pd.Timedelta(0):
            return False

        return True


class DataTimeSeries:
    def __init__(self, dict_df: Dict[Any, pd.DataFrame] = None, tsnames: str = None,
                 time_config:TimeConfig = None):
        
        self.tsnames = tsnames
        self.dict_df = dict_df if dict_df is not None else {}
        
        scenario_dimension = time_config.scenario_dimension

        self.dict_df = {
            scenario: self._create_empty_df(tsnames, scenario_dimension)
            for scenario in time_config.scenarios
        }

    @staticmethod
    def _create_empty_df(tsnames, n_rows):
        return pd.DataFrame(np.nan, index=range(n_rows), columns=tsnames, dtype=float)
    
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
            df.to_csv(f"{prefix}_key{key_str}.csv", index_label="timestep")

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
            dict_df[key] = pd.read_csv(file, index_col="timestep")

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
                 tsdata: DataTimeSeries = None, time_config: TimeConfig = None) -> None:
        """ProblemData is a class that represents the data of the RB-DNEP problem.

        Parameters
        ----------
        lecs : Dict[int, DataLEC], optional
            A dictionary that maps the index of the LEC to the DataLEC object.
        host_grid : ps.DataPowerSystem, optional
            The host grid of the problem.
        tsdata : Dict[int, ps.DataTimeSeries], optional
            A dictionary that maps the index of the LEC to the DataTimeSeries object.
        time_config : TimeConfig, optional
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
        sys_dict = {}

        # Write host grid
        sys_dict["host_grid"] = self.host_grid.to_dict()

        # Write LECs
        sys_dict["lecs"] = {idx: lec.to_dict() for idx, lec in self.lecs.items()}

        if self.time_config is not None:
            sys_dict["time_config"] = self.time_config.to_dict()

        return sys_dict

    def __str__(self):
        pretty_str = "Problem data:\n-------------\n\n"
        pretty_str += pretty_print_dict(self.system_dict())
        pretty_str += "\nTime series data:\n-----------------\n\n"
        pretty_str += str(self.tsdata)
        return pretty_str
    
    def __repr__(self):
        return self.__str__()

    def write(self, yaml_path: str, tsdata_path: str = None, format: str = "yaml") -> None:
        """Write the problem data to a YAML file.

        Parameters:
        -----------
        - yaml_path: str, the path to the YAML file.
        - tsdata_path: str, the path to the time series data file.
        """
        yaml_data = {}

        # Write host grid
        yaml_data["host_grid"] = self.host_grid.to_dict()

        # Write LECs
        yaml_data["lecs"] = {idx: lec.to_dict() for idx, lec in self.lecs.items()}

        # Write time configuration if it exists
        if self.time_config is not None:
            yaml_data["time_config"] = self.time_config.to_dict()

        # Ensure the yaml_path has the correct extension if not already present
        if not yaml_path.lower().endswith('.' + format):
            yaml_path += '.' + format

        # Serialize the data to a YAML file
        with open(yaml_path, "w") as file:
            yaml.dump(yaml_data, file, sort_keys=False)

        # Write time series data if it exists
        if self.tsdata:
            self.tsdata.write(tsdata_path)

    @classmethod
    def read(cls, yaml_path: str, tsdata_path: str = None) -> 'ProblemData':
        """Read the problem data from a YAML file.

        Parameters:
        -----------
        - yaml_path: str, the path to the YAML file.
        - tsdata_path: str, the path to the time series data file.
        """
        with open(yaml_path, "r") as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)


        # Read host grid
        host_grid = ps.DataPowerSystem.from_dict(yaml_data["host_grid"])

        # Read LECs
        lecs = {int(idx): DataLEC.from_dict(lec)
                for idx, lec in yaml_data["lecs"].items()}

        # Read time configuration if it exists
        if "time_config" in yaml_data:
            time_config = TimeConfig.from_dict(yaml_data["time_config"])
        else:
            time_config = None

        # Read time series data if it exists
        if tsdata_path is not None:
            tsdata = ps.DataTimeSeries.read(tsdata_path)
        else:
            tsdata = None

        return cls(lecs=lecs, host_grid=host_grid, tsdata=tsdata,
                time_config=time_config)

    def set_default_tsnames(self):
        
        l_demands_p_tsnames = []
        for demand in self.host_grid.demands.values():
            p_tsname = f"ds_p_MW_{demand.index}"
            q_tsname = f"ds_q_MVAr_{demand.index}"
            demand.p_MW.set_tsname(p_tsname)
            l_demands_p_tsnames.append(p_tsname)
            demand.q_MVAr.set_tsname(q_tsname)

        for generator in self.host_grid.generators.values():
            generator.pmax_MW.set_tsname(f"ds_p_MW_{generator.index}")

        for lec_idx, lec in self.lecs.items():
            for demand in lec.demands.values():
                demand.p_MW.set_tsname(f"lec{lec_idx}_p_MW_{demand.index}")
                demand.q_MVAr.set_tsname(f"lec{lec_idx}_q_MVAr_{demand.index}")

            for generator in lec.generators.values():
                generator.pmax_MW.set_tsname(f"lec{lec_idx}_pmax_MW_{generator.index}")

            for invgenerator in lec.inv_generators.values():
                invgenerator.pmax_MW.set_tsname(f"lec_inv_{lec_idx}_pmax_MW_{generator.index}")


    def time_series_structure(self) -> List[str]:

        ts_structure = {
            "host_grid": {"Demand.p_MW": [], "Demand.q_MVAr": [], "Generator.pmax_MW": []},
            "lecs": {
                lec_index:
                {"Demand.p_MW": [], "Demand.q_MVAr": [], "Generator.pmax_MW": []} 
                for lec_index in self.lecs.keys()
            }
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

            for invgenerator in lec.inv_generators.values():
                ts_structure["lecs"][lec_idx]["Generator.pmax_MW"].append(invgenerator.pmax_MW.tsname)

        return ts_structure
    
    def help(self):
        ts_structure = self.time_series_structure()
        pretty_str = "Time series structure:\n----------------------\n\n"

        # Explanation for Host Grid data
        pretty_str += "Host Grid Time Series Data:\n"
        pretty_str += "  - Demand.p_MW: Active power demands in MW at different nodes of the host grid.\n"
        pretty_str += "  - Demand.q_MVAr: Reactive power demands in MVAr at different nodes of the host grid.\n"
        pretty_str += "  - Generator.pmax_MW: Maximum power output capabilities of generators in MW.\n\n"

        # Explanation for LECs data
        pretty_str += "LECs (Local Energy Communities) Time Series Data:\n"
        pretty_str += "  Each LEC has its own set of time series data:\n"
        pretty_str += "  - Demand.p_MW: Active power demands in MW at different nodes within the LEC.\n"
        pretty_str += "  - Demand.q_MVAr: Reactive power demands in MVAr at different nodes within the LEC.\n"
        pretty_str += "  - Generator.pmax_MW: Maximum power output capabilities of generators in MW within the LEC.\n\n"

        # Print the time series structure
        pretty_str += "This instance requires the following:\n"
        pretty_str += "-------------------------------------\n"

        pretty_str += pretty_print_dict(ts_structure)
        print(pretty_str)

    
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
            # tsnames.append(demand.q_MVAr.tsname)

        for generator in self.host_grid.generators.values():
            tsnames.append(generator.pmax_MW.tsname)

        # LECs
        for lec in self.lecs.values():
            for demand in lec.demands.values():
                tsnames.append(demand.p_MW.tsname)
                # tsnames.append(demand.q_MVAr.tsname)

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
