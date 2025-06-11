from pathlib import Path
from tame.agents.mappo import Agent, Args  # noqa: F401
from tame.envs.vmas_balance import Env

# Agent and Env
MAX_TS = 100


# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=2,
    seed=10,
    verbose=False,
    lr=0.001,
    episode_length=MAX_TS,
)

# Env args
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/balance")
RUN_NAME = f"mappo__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
CONT_ACTIONS = True
