#******************************************************************************#
#                                     SETS                                     #
#******************************************************************************#
param N_TIME_PERIODS;
set TIME_PERIODS := 1..N_TIME_PERIODS;
set SCENARIOS;
set K_t{TIME_PERIODS} within SCENARIOS;
set BUSES;
set LECS within BUSES;
set BUSES_NOLECS := BUSES diff LECS;
set BRANCHES;
set NEW_BRANCHES within BRANCHES;
set BRANCHES_NONEW := BRANCHES diff NEW_BRANCHES;
set L_t{TIME_PERIODS} within NEW_BRANCHES;
set REINFORCEMENT_OPTIONS;  # Reinforcement options
set RE_n{LECS} within REINFORCEMENT_OPTIONS;
set DGS;
set NEW_DGS;
set DGS_NONEW := DGS diff NEW_DGS;
set DG_n{LECS} within DGS;

set LOWER_LEVEL_VARS := {"z_F", "w_F", "p_FN", "p_G", "p_FS"};
set LOWER_LEVEL_CONS := {"p0_ll_linking", "p0_ll_budget", "p0_ll_pbalance",
    "p0_ll_pgbounds", "p0_ll_inv_logic_1", "p0_ll_inv_logic_2"};

#******************************************************************************#
#                                  PARAMETERS                                  #
#******************************************************************************#
param INFTY = 1e9;  # Float infinity definition
param BIG_M = 100000;  # Big-M constant
param B_L = 10000;  # Budget leader default value 10000
param B_F = 10000;  # Budget follower default value 10000
param C_S = 10000;  # Cost non-served energy default value 10000

param R{TIME_PERIODS} = 1;  # rate of return coefficient. default value 1

param C_LI{NEW_BRANCHES, TIME_PERIODS}, >= 0;
param C_LM{NEW_BRANCHES, TIME_PERIODS}, >= 0;

param C_FG{NEW_DGS, TIME_PERIODS}, >= 0;
param C_FM{NEW_DGS, TIME_PERIODS}, >= 0;

param LAMBDA_L{SCENARIOS}, >= 0;
param BETA_L{SCENARIOS}, >= 0;
param AUX_K2T{SCENARIOS}, integer, >= 0;  # Auxiliary mapping scenario to time.

param FROM_BUS{BRANCHES}, integer, >= 0;  # From bus (Source bus) of the branch 
param TO_BUS{BRANCHES}, integer, >= 0;  # To bus (Sink bus) of the branch
param Z{BRANCHES}, >= 0;
param MAX_FLOW{BRANCHES}, >= 0;

param REINFORCEMENT_CAPACITY{REINFORCEMENT_OPTIONS};  # Feeder capacity
param p_FN_0{LECS};  # Initial feeder capacity of LECs

param DEMAND_L{BUSES, SCENARIOS};
param DEMAND_F{LECS, SCENARIOS};
param PGMAX{DGS, SCENARIOS};

#******************************************************************************#
#                                  VARIABLES                                   #
#******************************************************************************#
# LEADER binary variables
var z_L{NEW_BRANCHES, TIME_PERIODS}, binary;
var w_L{NEW_BRANCHES, TIME_PERIODS}, binary;
var gamma{LECS, TIME_PERIODS}, binary;

# LEADER continuous variables
var p_LN{BUSES, SCENARIOS};
var p_LS{BUSES, SCENARIOS}, >= 0;
var losses_L{BRANCHES, SCENARIOS}, >= 0;
var f{BRANCHES, SCENARIOS};
var v{BUSES, SCENARIOS}, >= 0;

# FOLLOWER binary variables
var z_F{NEW_DGS, TIME_PERIODS}, binary;
var w_F{NEW_DGS, TIME_PERIODS}, binary;

# FOLLOWER continuous variables
var p_FN{LECS, SCENARIOS};
var p_G{DGS, SCENARIOS}, >= 0;
var p_FS{LECS, SCENARIOS};

#******************************************************************************#
#                                     MODEL                                    #
#******************************************************************************#
# problem0 upper-level objective
minimize p0_ul_objective:
    sum{t in TIME_PERIODS} (1/R[t]) *(
        # Investment terms
        + sum{l in L_t[t]} (C_LI[l, t] * z_L[l, t] + C_LM[l, t] * w_L[l, t])
        # Operational terms
        + sum{k in K_t[t]} (BETA_L[k] * (
            LAMBDA_L[k]* (sum{l in BRANCHES} (losses_L[l, k])
            - sum{n in BUSES}(p_LN[n, k]) ) + sum{n in BUSES}(C_S * p_LS[n, k]))
        )
    );

# problem0 lower-level objective (dummy constraint for representation purposes)
subject to LOWER_LEVEL_FOBJ:
    sum{t in TIME_PERIODS} (1/R[t]) * sum{g in NEW_DGS} (C_FG[g, t] * z_F[g,t])
    <= INFTY
;

# problem0 upper-level budget
subject to p0_ul_budget:
    sum{t in TIME_PERIODS, l in NEW_BRANCHES}(C_LI[l, t] * z_L[l, t]) <= B_L
;

# problem0 upper-level node balance 1
subject to p0_ul_node_balance_1{n in LECS, k in SCENARIOS}:
    # Power injection at node
    + p_LN[n, k] + p_FN[n, k] + p_LS[n, k]
    # Power inkection comming from branches
    + sum{l in BRANCHES: TO_BUS[l] == n} (f[l, k])
    - sum{l in BRANCHES: FROM_BUS[l] == n} (f[l, k])
    ==
    DEMAND_L[n, k]
;

# problem0 upper-level node balance 2
subject to p0_ul_node_balance_2{n in BUSES_NOLECS, k in SCENARIOS}:
    # Power injection at node
    + p_LN[n, k] + p_LS[n, k]
    # Power inkection comming from branches
    + sum{l in BRANCHES: TO_BUS[l] == n} (f[l, k])
    - sum{l in BRANCHES: FROM_BUS[l] == n} (f[l, k])
    ==
    DEMAND_L[n, k]
;

# problem0 upper-level power flow
subject to p0_ul_pf{l in BRANCHES_NONEW, k in SCENARIOS}:
    Z[l] * f[l, k] + v[TO_BUS[l], k] - v[FROM_BUS[l], k] == 0
;

# problem0 upper-level investment logic 1
subject to p0_ul_inv_logic_1{l in NEW_BRANCHES, k in SCENARIOS}:
    Z[l] * f[l, k] + v[TO_BUS[l], k] - v[FROM_BUS[l], k]
    <=
    BIG_M * (1 - w_L[l, AUX_K2T[k]])
;

# problem0 upper-level investment logic 2
subject to p0_ul_inv_logic_2{l in NEW_BRANCHES, k in SCENARIOS}:
    Z[l] * f[l, k] + v[TO_BUS[l], k] - v[FROM_BUS[l], k]
    >=
    BIG_M * (1 - w_L[l, AUX_K2T[k]])
;

# problem0 upper-level investment logic 3
subject to p0_ul_inv_logic_3{l in NEW_BRANCHES, t in TIME_PERIODS}:
    w_L[l, t] <= sum{tau in TIME_PERIODS: tau <= t} (z_L[l, t])
;

# problem0 upper-level logic reinforcement
subject to p0_ul_logic_reinfor{n in LECS, t in TIME_PERIODS}:
    sum{r in RE_n[n]} (gamma[r, t]) <= 1
;

#*************#
# Lower level #
#*************#

# problem0 lower-level linking constraints
subject to p0_ll_linking{n in LECS, k in SCENARIOS}:
    p_FN[n, k] - p_FN_0[n]
    <=
    sum{tau in 1..AUX_K2T[k]} sum{r in RE_n[n]} (gamma[n, AUX_K2T[k]])
;

# problem0 lower-level budget.
subject to p0_ll_budget:  
    sum{t in TIME_PERIODS, g in NEW_DGS} (C_FG[g, t] * z_F[g, t]) <= B_F
;

# problem0 lower-level power balance
subject to p0_ll_pbalance{n in LECS, k in SCENARIOS}:
    p_FN[n, k] == sum{g in DG_n[n]} (p_G[g, k]) + DEMAND_F[n, k] + p_FS[n, k]
;

# problem0 lower-level generated power bounds.
subject to p0_ll_pgbounds{g in DGS, k in SCENARIOS}:
    p_G[g, k] <= PGMAX[g, k]
;

# problem0 lower-level investment logic 1
subject to p0_ll_inv_logic_1{g in DGS, k in SCENARIOS}:
    p_G[g, k] <= BIG_M * w_F[g, AUX_K2T[k]]
;

# problem0 lower-level investment logic 2
subject to p0_ll_inv_logic_2 {g in NEW_DGS, t in TIME_PERIODS}:
    w_F[g, t] <= sum{tau in 1..t} (z_F[g, t])
;
