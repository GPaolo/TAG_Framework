from pettingzoo.sisl import multiwalker_v9


class Env:
    """A wrapper class for the PettingZoo multiwalker environment.

    This class provides a simplified interface to the multiwalker environment from PettingZoo,
    maintaining core functionality while allowing for customization of key parameters.

    Args:
        total_agents (int, optional): Number of walker agents in the environment. Defaults to 4.
        max_ts (int, optional): Maximum number of timesteps per episode. Defaults to 500.
        render_mode (str, optional): Rendering mode for the environment. Defaults to "rgb_array".
        continuous_actions (bool, optional): Flag to determine if actions are continuous. Defaults to False.

    Attributes:
        max_ts (int): Maximum number of timesteps per episode
        total_agents (int): Number of walker agents in the environment
        env (PettingZoo.ParallelEnv): The underlying multiwalker environment instance

    Methods:
        step(action, **kwargs): Takes a step in the environment using the provided action.
            Returns the next observation, reward, done flag, and info dictionary.

    Note:
        The class implements a __getattr__ method to provide direct access to the underlying
        environment's attributes and methods that are not explicitly defined in this wrapper.
    """

    def __init__(
        self,
        total_agents: int = 4,
        max_ts: int = 500,
        render_mode: str = "rgb_array",
        continuous_actions: bool = False,
    ):
        self.max_ts = max_ts
        self.total_agents = total_agents
        self.env = multiwalker_v9.parallel_env(
            render_mode=render_mode,
            n_walkers=self.total_agents,
            terrain_length=75,
            max_cycles=self.max_ts,
            shared_reward=False,
        )

    def step(self, action, **kwargs):
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)
