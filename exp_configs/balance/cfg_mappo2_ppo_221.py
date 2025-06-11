from pathlib import Path
from tame.agents.mappo2_ppo_cont import Agent, Args  # noqa: F401
from tame.envs.vmas_balance import Env

# Agent and Env
MAX_TS = 100


# Agent args
agent_args = Args(
    total_timesteps=2000000,
    cuda=3,
    seed=10,
    verbose=False,
    episode_length=MAX_TS,
    ifreq_top=2,
    ifreq_mid=2,
)

# Env args
TOTAL_AGENTS = 4

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/balance")
RUN_NAME = f"mappo2_ppo_cont_221__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
