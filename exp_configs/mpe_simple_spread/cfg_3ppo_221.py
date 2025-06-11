from pathlib import Path
from tame.agents.ppo3 import Agent, Args  # noqa: F401
from tame.envs.mpe_simple_spread import Env  # noqa: F401

# Agent and Env
MAX_TS = 100


# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=2,
    seed=10,
    verbose=False,
    ifreq_top=2,
    ifreq_mid=2,
    learn_comm=False,
    save_all_trace=False,
)

# Env args
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/simple_spread")
RUN_NAME = f"3ppo_221__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
