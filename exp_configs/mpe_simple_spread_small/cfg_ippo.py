from pathlib import Path

# Agent and Env
from tame.agents.i_ppo import Agent, Args
from tame.envs.mpe_simple_spread import SmallObsEnv as Env

# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=2,
    seed=10,
    verbose=False,
)

# Env args
MAX_TS = 100
TOTAL_AGENTS = 1

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/simple_spread_small")
RUN_NAME = f"ippo__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
