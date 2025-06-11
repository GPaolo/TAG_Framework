from pettingzoo.mpe import simple_spread_v3
from gymnasium import spaces

import pygame
import numpy as np


class Env:
    """A wrapper environment for the simple_spread_v3 environment from PettingZoo MPE.

    This environment wraps the simple spread environment to allow for custom colors and
    maintains the color scheme across resets.

    Args:
        total_agents (int, optional): Number of agents in the environment. Defaults to 4.
        max_ts (int, optional): Maximum number of timesteps per episode. Defaults to 100.
        render_mode (str, optional): Rendering mode for the environment. Defaults to "rgb_array".
        colors (dict or None, optional): Dictionary mapping agent indices to RGB color values (as lists).
            Defaults to alternating between red [1,0,0] and blue [0,0,1] for up to 10 agents.
        continuous_actions (bool, optional): Whether to use continuous action space. Defaults to False.

    Attributes:
        env (PettingZoo.ParallelEnv): The underlying simple spread environment.
        colors (dict): The color scheme for the agents.
        max_ts (int): Maximum timesteps per episode.
        total_agents (int): Number of agents in the environment.

    Methods:
        reset(): Resets the environment and maintains the color scheme.
        step(action): Takes a step in the environment.
    """

    def __init__(
        self,
        total_agents: int = 4,
        max_ts: int = 100,
        render_mode: str = "rgb_array",
        colors: None | dict = {
            0: [1, 0, 0],
            1: [0, 0, 1],
            2: [1, 0, 0],
            3: [0, 0, 1],
            4: [1, 0, 0],
            5: [0, 0, 1],
            6: [1, 0, 0],
            7: [0, 0, 1],
            8: [1, 0, 0],
            9: [0, 0, 1],
        },
        continuous_actions: bool = False,
    ):
        pygame.init()
        self.max_ts = max_ts
        self.total_agents = total_agents
        self.env = simple_spread_v3.parallel_env(
            N=total_agents,
            local_ratio=0.5,
            max_cycles=max_ts,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        self.colors = colors

    def reset(self):
        """Reset the environment.

        This method resets the environment and optionally updates agent colors. It's needed because
        the environment also resets the colors during its own reset.

        Returns:
            tuple: A tuple containing:
                - obs: Initial observations for each agent
                - infos: Additional information dictionary for each agent

        Note:
            If self.colors is set, it will override the default colors for each agent
            in the environment after reset.
        """
        obs, infos = self.env.reset()
        if self.colors is not None:
            for i, agent in enumerate(self.env.unwrapped.world.agents):
                agent.color = np.array(self.colors[i])
        return obs, infos

    def step(self, action, **kwargs):
        """
        Execute one time step within the environment.

        Args:
            action: Action to be executed in the environment
            **kwargs: Additional keyword arguments

        Returns:
            tuple: Returns the step information from the environment, typically including:
                - observation (object): Agent's observation of the current environment
                - reward (float): Amount of reward returned after previous action
                - done (bool): Whether the episode has ended
                - info (dict): Contains auxiliary diagnostic information
        """
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)


class SmallObsEnv(Env):
    """
    A modified version of the simple spread environment with a reduced observation space.
    Each agent only observes its own state and the relative positions of landmarks,
    without direct observation of other agents.

    Args:
        total_agents (int): Number of agents in the environment. Defaults to 4.
        max_ts (int): Maximum number of timesteps per episode. Defaults to 100.
        render_mode (str): Rendering mode for visualization. Defaults to "rgb_array".
        colors (dict or None): Color mapping for agents and landmarks. Alternates between red [1,0,0]
                              and blue [0,0,1] for up to 10 agents by default.
        continuous_actions (bool): If True, actions are continuous. If False, actions are discrete.
                                 Defaults to False.

    Observation Space:
        The observation space for each agent consists of:
        - Agent's velocity (2D)
        - Agent's position (2D)
        - Relative positions of landmarks (2D * number_of_agents)
        Total size = 4 + (total_agents * 2)

    Note:
        Unlike the parent environment, agents cannot observe other agents' positions or velocities,
        making this a partially observable environment focused only on landmark positions.
    """

    def __init__(
        self,
        total_agents: int = 4,
        max_ts: int = 100,
        render_mode: str = "rgb_array",
        colors: None | dict = {
            0: [1, 0, 0],
            1: [0, 0, 1],
            2: [1, 0, 0],
            3: [0, 0, 1],
            4: [1, 0, 0],
            5: [0, 0, 1],
            6: [1, 0, 0],
            7: [0, 0, 1],
            8: [1, 0, 0],
            9: [0, 0, 1],
        },
        continuous_actions: bool = False,
    ):
        super().__init__(
            total_agents=total_agents,
            max_ts=max_ts,
            render_mode=render_mode,
            colors=colors,
            continuous_actions=continuous_actions,
        )
        # [self_vel, self_pos, landmark_rel_positions]
        self.observation_size = 4 + self.total_agents * 2

    @property
    def observation_spaces(self):
        """Gets the observation spaces of the environment.

        Returns:
            Dict[str, spaces.Box]
                A dictionary mapping agent IDs to their corresponding observation spaces.
                Each observation space is a Box with shape (observation_size,) and unbounded values.
                The observation spaces are cached after first computation.
        """
        if not hasattr(self, "_observation_spaces"):
            self._observation_spaces = {
                agent: spaces.Box(
                    low=-np.float32(np.inf),
                    high=+np.float32(np.inf),
                    shape=(self.observation_size,),
                    dtype=np.float32,
                )
                for agent in self.env.action_spaces
            }
        return self._observation_spaces

    def _modify_obs(self, observation):
        """
        Modifies the observation by selecting a subset of the original observation.

        Args:
            observation (dict): Dictionary containing observations for each agent, where each observation
                               is an array-like object.

        Returns:
            dict: Modified observation dictionary where each agent's observation is truncated to
                  self.observation_size elements.

        Example:
            If observation = {'agent_0': [1,2,3,4,5], 'agent_1': [6,7,8,9,10]}
            and self.observation_size = 3
            Returns {'agent_0': [1,2,3], 'agent_1': [6,7,8]}
        """
        small_obs = {}
        for agent, obs in observation.items():
            small_obs[agent] = obs[: self.observation_size]
        return small_obs

    def reset(self):
        """
        Reset the environment to an initial state.

        This method extends the parent class reset by modifying the observation
        before returning it.

        Returns:
            tuple: A tuple containing:
                - modified observation (type depends on environment)
                - info dictionary with additional information
        """
        obs, info = super().reset()
        return self._modify_obs(obs), info

    def step(self, action, **kwargs):
        """
        Executes one time step within the environment.

        This method extends the parent class's step method by modifying the observation
        after each step.

        Args:
            action: Action to be executed in the environment
            **kwargs: Additional keyword arguments to be passed to the parent class's step method

        Returns:
            tuple: A tuple containing:
                - modified observation (object): Agent's modified observation of the current environment
                - reward (float): Amount of reward returned after previous action
                - terminated (bool): Whether the episode has ended
                - truncated (bool): Whether the episode was artificially terminated
                - info (dict): Contains auxiliary diagnostic information
        """
        obs, r, ter, trun, inf = super().step(action, **kwargs)
        return self._modify_obs(obs), r, ter, trun, inf
