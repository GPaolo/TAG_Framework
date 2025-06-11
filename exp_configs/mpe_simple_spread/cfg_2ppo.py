from pathlib import Path
from tame.agents.ppo2 import Agent, Args  # noqa: F401
from tame.envs.mpe_simple_spread import Env  # noqa: F401

# Agent and Env
MAX_TS = 100


# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=0,
    seed=8,
    verbose=False,
)

# Env args
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/simple_spread")
RUN_NAME = f"2ppo__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
