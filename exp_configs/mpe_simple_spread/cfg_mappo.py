from pathlib import Path
from tame.agents.mappo import Agent, Args  # noqa: F401
from tame.envs.mpe_simple_spread import Env  # noqa: F401

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
    use_rnn=True,
    rnn_hidden_dim=64,
)

# Env args
TOTAL_AGENTS = 3

# Exp args
EVAL_RUNS = 1
SAVE_PATH = Path("./exps/simple_spread")
RUN_NAME = f"mappo__a{TOTAL_AGENTS}_s{agent_args.seed}"
TRAIN = True
CONT_ACTIONS = True
