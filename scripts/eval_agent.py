import click

from tame.utils.config import load_config
from tame.utils.utils import evaluate


@click.command()
@click.option(
    "--config-path",
    type=click.Path(),
)
@click.option(
    "--save-path",
    type=click.Path(),
    default=None,
)
def eval_agent(config_path, save_path):
    config = load_config(config_path=config_path)
    config.TRAIN = False  # type: ignore

    # Setup
    # ---------------------------
    env = config.Env(total_agents=config.TOTAL_AGENTS, max_ts=config.MAX_TS)  # type: ignore
    agent = config.Agent(env=env, args=config.agent_args)
    # ---------------------------
    if save_path is None:
        save_path = config.SAVE_PATH / config.RUN_NAME

    agent.load_agent(load_path=save_path / "training")

    # Evaluate
    eval_env = config.Env(total_agents=config.TOTAL_AGENTS, max_ts=config.MAX_TS)  # type: ignore
    evaluate(
        agent=agent,
        env=eval_env,
        eval_runs=config.EVAL_RUNS,
        save_path=config.SAVE_PATH / config.RUN_NAME,
    )


if __name__ == "__main__":
    eval_agent()
