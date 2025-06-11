from pathlib import Path
from tame.agents.mappo2_ppo_cont import Agent, Args  # noqa: F401
from tame.envs.mpe_simple_spread import Env  # noqa: F401

# Agent and Env
MAX_TS = 100


# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=2,
    seed=10,
    verbose=False,
    episode_length=MAX_TS,
)

# Env args
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/simple_spread")
RUN_NAME = f"mappo2_ppo_cont__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
