from dataclasses import dataclass, field
import numpy as np

import shilps.powersystems as ps
from rb_dnep.problem_data import ProblemData


class DDNEPModel:

    @dataclass(slots=True)
    class DATA_DDNEP:
        """A data class for the deterministic DNEP."""
        # Sets
        N_TIME_PERIODS: int = 0
        TIME_PERIODS: list = field(default_factory=list)
        SCENARIOS: list = field(default_factory=list)
        K_t: dict = field(default_factory=dict)
        BUSES: list = field(default_factory=list)
        LECS: list = field(default_factory=list)
        BUSES_NOLECS: list = field(default_factory=list)
        BRANCHES: list = field(default_factory=list)
        NEW_BRANCHES: list = field(default_factory=list)
        BRANCHES_NONEW: list = field(default_factory=list)
        L_t: dict = field(default_factory=dict)
        REINFORCEMENT_OPTIONS: list = field(default_factory=list)
        RE_n: dict = field(default_factory=dict)
        DGS: list = field(default_factory=list)
        NEW_DGS: list = field(default_factory=list)
        DGS_NONEW: list = field(default_factory=list)
        DG_n: dict = field(default_factory=dict)

        # Non-indexed parameters
        INFTY: float = 1e9
        BIG_M: int = 100000
        B_L: int = 10000
        B_F: int = 10000
        C_S: int = 10000

        # Indexed parameters
        R: np.ndarray = field(default_factory=lambda: np.array([]))
        C_LI: np.ndarray = field(default_factory=lambda: np.array([]))
        C_LM: np.ndarray = field(default_factory=lambda: np.array([]))
        C_FG: np.ndarray = field(default_factory=lambda: np.array([]))
        C_FM: np.ndarray = field(default_factory=lambda: np.array([]))
        LAMBDA_L: np.ndarray = field(default_factory=lambda: np.array([]))
        BETA_L: np.ndarray = field(default_factory=lambda: np.array([]))
        AUX_K2T: np.ndarray = field(default_factory=lambda: np.array([]))
        FROM_BUS: np.ndarray = field(default_factory=lambda: np.array([]))
        TO_BUS: np.ndarray = field(default_factory=lambda: np.array([]))
        Z: np.ndarray = field(default_factory=lambda: np.array([]))
        MAX_FLOW: np.ndarray = field(default_factory=lambda: np.array([]))
        REINFORCEMENT_CAPACITY: np.ndarray = field(default_factory=lambda: np.array([]))
        p_FN_0: np.ndarray = field(default_factory=lambda: np.array([]))
        DEMAND_L: np.ndarray = field(default_factory=lambda: np.array([]))
        DEMAND_F: np.ndarray = field(default_factory=lambda: np.array([]))
        PGMAX: np.ndarray = field(default_factory=lambda: np.array([]))


    def __init__(self, pdata: ProblemData,  config=None):
        self.config = config
        self.pdata = pdata
        self.data_dat = None

    def parse_data_to_dat(self):
        """Parse the data to a .dat file. Creates a DATA_DDNEP instance."""
        self.data_dat = self.DATA_DDNEP()

        # Sets
        N_TIME_PERIODS = self.pdata.time_config.n_periods
        TIME_PERIODS = list(range(1, self.pdata.time_config.n_periods + 1))
        SCENARIOS = list(range(1, self.pdata.time_config.n_periods + 1))
        K_t = {t: 1 for t in self.data_dat.TIME_PERIODS}
        BUSES = [bus.index for bus in self.pdata.host_grid.buses.values()]
        LECS = [lec.index for lec in self.pdata.lecs.values()]
        BUSES_NOLECS = [bus.index for bus in self.pdata.host_grid.buses.values() if bus.index not in self.data_dat.LECS]
        BRANCHES = [branch.index for branch in self.pdata.host_grid.branches.values()]
        NEW_BRANCHES = [branch.index for branch in self.pdata.host_grid.branches.values() if branch.index not in self.data_dat.BRANCHES]
        BRANCHES_NONEW = [branch.index for branch in self.pdata.host_grid.branches.values() if branch.index not in self.data_dat.NEW_BRANCHES]
        L_t = {t: 1 for t in self.data_dat.TIME_PERIODS}
        REINFORCEMENT_OPTIONS = [1, 2, 3]
        RE_n = {n: 1 for n in self.data_dat.REINFORCEMENT_OPTIONS}
        DGS = [dg.index for dg in self.pdata.host_grid.dgs.values()]
        NEW_DGS = [dg.index for dg in self.pdata.host_grid.dgs.values() if dg.index not in self.data_dat.DGS]
        DGS_NONEW = [dg.index for dg in self.pdata.host_grid.dgs.values() if dg.index not in self.data_dat.NEW_DGS]
        DG_n = {n: 1 for n in self.data_dat.DGS}

        
        R = np.array([branch.resistance for branch in self.pdata.host_grid.branches.values()])
        C_LI = np.array([branch.cost_line_installation for branch in self.pdata.host_grid.branches.values()])
        C_LM = np.array([branch.cost_line_maintenance for branch in self.pdata.host_grid.branches.values()])
        C_FG = np.array([dg.cost_fuel_gas for dg in self.pdata.host_grid.dgs.values()])
        C_FM = np.array([dg.cost_fuel_maintenance for dg in self.pdata.host_grid.dgs.values()])
        LAMBDA_L = np.array([lec.lambda_l for lec in self.pdata.lecs.values()])
        BETA_L = np.array([lec.beta_l for lec in self.pdata.lecs.values()])
        AUX_K2T = np.array([self.config.aux_k2t for _ in self.data_dat.TIME_PERIODS])
        FROM_BUS = np.array([branch.from_bus.index for branch in self.pdata.host_grid.branches.values()])
        TO_BUS = np.array([branch.to_bus.index for branch in self.pdata.host_grid.branches.values()])
        Z = np.array([branch.impedance for branch in self.pdata.host_grid.branches.values()])
        MAX_FLOW = np.array([branch.max_flow for branch in self.pdata.host_grid.branches.values()])
        REINFORCEMENT_CAPACITY = np.array([self.config.reinforcement_capacity for _ in self.data_dat.REINFORCEMENT_OPTIONS])
        p_FN_0 = np.array([dg.p_fn_0 for dg in self.pdata.host_grid.dgs.values()])
        DEMAND_L = np.array([lec.demand_l for lec in self.pdata.lecs.values()])
        DEMAND_F = np.array([dg.demand_f for dg in self.pdata.host_grid.dgs.values()])
        PGMAX = np.array([dg.pgmax for dg in self.pdata.host_grid.dgs.values()])
    