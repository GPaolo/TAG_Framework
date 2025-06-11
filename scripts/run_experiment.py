from pathlib import Path

import click
import torch
from gymnasium.spaces import Dict as GymDict

from tame.hierarchy.base_agent import BaseAgent, LevelAgent
from tame.utils.config import load_config, save_config
from tame.utils.utils import evaluate, hasmethod


@click.command()
@click.option(
    "--config-path",
    type=click.Path(),
)
@click.option(
    "--config-name",
    type=str,
)
@click.option("--n-agents", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--cuda", type=int, default=None)
@click.option("--training-steps", type=int, default=None)
@click.option("--train-agent", type=bool, default=None)
@click.option("--exp-name", type=str, default=None)
def run_experiment(
    config_path,
    config_name,
    n_agents,
    seed,
    cuda,
    training_steps,
    train_agent,
    exp_name,
):
    config_path = Path(config_path)
    if ".py" not in config_name:
        config_name = f"{config_name}.py"
    print(f"Loading config from {config_path / config_name}")
    config = load_config(config_path=config_path / config_name)

    # This is to get the new exp_name
    if exp_name is not None:
        config.RUN_NAME = exp_name  # type: ignore
    if exp_name is None:
        split_name = [
            f"s{seed}"
            if (x == f"s{config.agent_args.seed}" and seed is not None)
            else f"a{n_agents}"
            if (x == f"a{config.TOTAL_AGENTS}" and n_agents is not None)
            else x
            for x in config.RUN_NAME.split("_")
        ]
        config.RUN_NAME = "_".join(split_name)  # type: ignore

    if n_agents is not None:
        config.TOTAL_AGENTS = n_agents
    if seed is not None:
        config.agent_args.seed = seed
    if cuda is not None:
        config.agent_args.cuda = cuda
    if training_steps is not None:
        config.agent_args.total_timesteps = training_steps
    if train_agent is not None:
        config.TRAIN = train_agent

    print(f"Saving config to exp dir {config.SAVE_PATH / config.RUN_NAME}")
    save_config(
        config=config,
        save_path=config.SAVE_PATH / config.RUN_NAME,
        original_path=config_path / config_name,
    )

    # # Setup
    # ---------------------------
    try:
        env = config.Env(
            total_agents=config.TOTAL_AGENTS,  # type: ignore
            max_ts=config.MAX_TS,  # type: ignore
            continuous_actions=config.CONT_ACTIONS,  # type: ignore
        )
    except Exception as e:
        env = config.Env(total_agents=config.TOTAL_AGENTS, max_ts=config.MAX_TS)  # type: ignore

    if issubclass(config.Agent, LevelAgent):
        if config.agent_args.cuda >= 0:
            device = torch.device(
                f"cuda:{config.agent_args.cuda}" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = "cpu"

        agent = config.Agent(
            observation_space=GymDict(env.observation_spaces),
            action_space=GymDict(env.action_spaces),
            communication_space=None,
            device=device,
            args=config.agent_args,
        )
    elif issubclass(config.Agent, BaseAgent):
        agent = config.Agent(env=env, args=config.agent_args)
    else:
        raise ValueError(
            "Agent type is not supported! Only BaseAgent or LevelAgent are supported"
        )
    # ---------------------------
    # Train
    if config.TRAIN:
        agent.train(env=env, log_path=config.SAVE_PATH, run_name=config.RUN_NAME)
    else:
        agent.load_agent(load_path=config.SAVE_PATH / config.RUN_NAME)

    # Evaluate
    if hasmethod(agent, "reset"):
        agent.reset()  # type: ignore

    eval_env = config.Env(total_agents=config.TOTAL_AGENTS, max_ts=1000)  # type: ignore
    evaluate(
        agent=agent,
        env=eval_env,
        eval_runs=config.EVAL_RUNS,
        save_path=config.SAVE_PATH / config.RUN_NAME,
    )


if __name__ == "__main__":
    run_experiment()
