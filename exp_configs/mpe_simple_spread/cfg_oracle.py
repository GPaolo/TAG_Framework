from pathlib import Path
from scripts.run_experiment import ROOT

# Agent and Env
from tame.envs.mpe_simple_spread import Oracle as Agent
from tame.envs.mpe_simple_spread import Env, OracleArgs

# Agent args
agent_args = OracleArgs(total_timesteps=1000, seed=1, cuda=0, verbose=True)

# Env args
MAX_TS = 100
TOTAL_AGENTS = 10

# Exp Args
EVAL_RUNS = 1
SAVE_PATH = ROOT / Path("./exps/simple_spread")
RUN_NAME = f"oracle__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
