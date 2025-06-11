from pathlib import Path

# Agent and Env
from tame.agents.monolithic_ppo import Agent, Args
from tame.envs.vmas_balance import Env

# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=2,
    seed=10,
    verbose=False,
)

# Env args
MAX_TS = 100
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/balance")
RUN_NAME = f"monolithic_ppo__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
