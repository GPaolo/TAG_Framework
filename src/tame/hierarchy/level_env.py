from tame.data_handling.trace import Trace, RewardTrace
from tame.utils.utils import hasmethod
from tame.hierarchy.base_agent import LevelAgent
from typing import Tuple, Any, Dict
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import random
from gymnasium.spaces import Dict as GymDict
import json
import os
from typing import TypeGuard

trace_types = {"none": None, "full": Trace, "reward": RewardTrace}

# TODO add checks for action spaces types of agents (make sure that discrete only agents do not go on continuous envs)


def is_nested_dict(
    obj: Dict[str, np.ndarray] | Dict[str, Dict[str, np.ndarray]],
) -> TypeGuard[Dict[str, Dict[str, np.ndarray]]]:
    """Check if a dictionary is a nested dictionary.

    This function determines if a dictionary contains dictionaries as values.
    It checks whether all values in the input dictionary are themselves dictionaries.

    Args:
        obj: A dictionary that contains either numpy arrays or dictionaries of numpy arrays.
            Can be either Dict[str, np.ndarray] or Dict[str, Dict[str, np.ndarray]].

    Returns:
        TypeGuard[Dict[str, Dict[str, np.ndarray]]]: True if all values in the dictionary
            are dictionaries, False otherwise. The TypeGuard ensures type narrowing
            in a type checker.

    Examples:
        >>> d1 = {"a": {"x": np.array([1])}}  # Nested dict
        >>> d2 = {"a": np.array([1])}         # Flat dict
        >>> is_nested_dict(d1)
        True
        >>> is_nested_dict(d2)
        False
    """
    return all(isinstance(v, dict) for v in obj.values())


class LevelEnv:
    """A hierarchical environment wrapper that manages multiple agents at a specific level.

    This class creates a hierarchical layer of agents that can interact with both higher and lower levels
    in a hierarchical reinforcement learning setup. It handles the communication between levels,
    action execution, and agent management.

    Attributes:
        verbose (int): Verbosity level for logging.
        concat_obs (bool): Whether to concatenate top observations with lower messages.
        obs_from_bottom (dict): Storage for observations from lower level.
        agents (Dict[str, LevelAgent]): Dictionary of agents at this level.
        env (Any): The lower level environment.
        uplinks (Dict[str, list]): Mapping of higher level agents to their controlled agents.
        downlinks (Dict[str, list]): Mapping of agents to their controlled lower level agents.
        action_ifreq (int): Action frequency relative to lower level steps.
        trace (object): Trace object for logging episode data.
        level_ts (int): Total timesteps at this level.
        episode_ts (int): Timesteps in current episode.
        episode_idx (int): Current episode index.
        name (str): Identifier for this level.
        logger (SummaryWriter): Logger for metrics.
        save_path (Path): Path for saving model data.
        observation_spaces (GymDict): Observation spaces for each agent.
        action_spaces (GymDict): Action spaces for each agent.
        episode_returns (dict): Cumulative returns for current episode.
        env_actions (dict): Latest actions sent to lower level.

    Methods:
        action_space(agent_name): Returns the action space for a specific agent.
        observation_space(agent_name): Returns the observation space for a specific agent.
        seed(seed): Sets random seeds for reproducibility.
        set_logger(logger, save_path): Configures logging and saving directories.
        reset(): Resets the level and returns initial observations.
        step(action, training): Executes one step of the environment.
        update_step(): Performs learning updates for agents.
        save(save_path): Saves all agents' models.
        load(load_path): Loads all agents' models.
        make_message(...): Creates messages for higher level from lower level data.
        act(observation, training): Generates actions for lower level from agents.
    """

    def __init__(
        self,
        agents: Dict[str, LevelAgent],
        uplinks: Dict[str, list] | None,
        downlinks: Dict[str, list],
        env: Any,
        action_space: GymDict | None = None,  # {str: Box | Discrete}
        concat_obs: bool = False,
        trace_type: str = "none",
        name: str = "level_0",
        verbose: int = 0,
        action_ifreq: int = 1,
    ):
        """This class wraps a level of agents to behave like a gym environment.

        Manages hierarchical interactions between agents at different levels by handling observations,
        actions and message passing between levels.

        Args:
            agents (Dict[str, LevelAgent]): Dictionary of agents at this level, keyed by agent name
            uplinks (Dict[str, list] | None): Mapping of how each agent links to higher level agents. Format: {higher_agent: [lower_agents]}
            downlinks (Dict[str, list]): Mapping of how each agent links to lower level agents. Format: {agent: [lower_agents]}
            env (Any): The environment this level interacts with (typically lower level env)
            action_space (GymDict | None, optional): Custom action space. Only valid if concat_obs=True. Defaults to None.
            concat_obs (bool, optional): If True, concatenates top observations with lower messages [msg, obs].
                                       If False, agent only sees top observations. Defaults to False.
            trace_type (str, optional): Type of execution trace to maintain. Must be one of trace_types. Defaults to "none".
            name (str, optional): Name identifier for this level. Defaults to "level_0".
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            action_ifreq (int, optional): Action interval frequency relative to lower level steps.
                                         Minimum value is 1. Defaults to 1.

            ValueError: If trace_type is invalid or if action_space is provided without concat_obs=True

        Attributes:
            verbose (int): Verbosity level
            concat_obs (bool): Whether to concatenate observations
            obs_from_bottom (dict): Observations received from lower level
            agents (dict): The level's agents
            env (Any): Lower level environment
            uplinks (dict): Upper level connections
            downlinks (dict): Lower level connections
            action_ifreq (int): Action interval frequency
            trace: Trace object for logging execution
            level_ts (int): Global timestep counter (not reset between episodes)
            episode_ts (int): Episode timestep counter (reset each episode)
            episode_idx (int): Current episode index
            name (str): Level name identifier
            logger: Logger object
            save_path: Path for saving data
            observation_spaces: Observation spaces for the level
            action_spaces: Action spaces for the level"""
        self.verbose = verbose
        self.concat_obs = concat_obs
        self.obs_from_bottom = {}
        self.agents = agents
        self.env = env
        self.uplinks = uplinks
        self.downlinks = downlinks
        self.action_ifreq = max(1, action_ifreq)
        # Semi-MDP bookkeeping for levels acting every action_ifreq lower-level steps:
        # one transition per decision, with the reward accumulated over the interval.
        self._decision_obs: Dict[str, Any] = {}
        self._interval_rew: Dict[str, float] = {}
        self._interval_done: Dict[str, bool] = {}
        self._interval_stored: Dict[str, bool] = {}
        if trace_type not in trace_types:
            raise ValueError(
                f"Wrong trace type specified for {name}. Given: {trace_type} - Available: {trace_types.keys()}"
            )
        self.trace = trace_types[trace_type]
        if self.trace is not None:
            self.trace = self.trace()
        self.level_ts = 0  # This one is not reset
        self.episode_ts = 0  # This one is reset
        self.episode_idx = -1
        self.name = name

        self.logger = None
        self.save_path = None
        self.observation_spaces = self._make_observation_space()

        if action_space is not None:
            if not self.concat_obs:
                raise ValueError(
                    f"{name} - You can only specify an action space if you pass concat_obs=True"
                )
            self.action_spaces = action_space
        else:
            self.action_spaces = self._make_action_space()
        print(f"Created level {self.name} with agents: {list(self.downlinks.keys())}")

    def action_space(self, agent_name):
        """
        Get the action space for a specific agent.

        Args:
            agent_name (str): The name of the agent whose action space is to be retrieved.

        Returns:
            gym.Space: The action space associated with the specified agent.
        """
        return self.action_spaces[agent_name]

    def observation_space(self, agent_name):
        """Get the observation space for a specific agent.

        This method returns the observation space associated with the given agent name.
        The observation space defines the structure and bounds of observations that the
        agent can receive from the environment.

        Args:
            agent_name (str): The name of the agent whose observation space is requested.

        Returns:
            gym.Space: The observation space for the specified agent.

        Raises:
            KeyError: If the agent_name is not found in the observation_spaces dictionary.
        """
        return self.observation_spaces[agent_name]

    def _make_observation_space(self) -> GymDict:  # {str: Box | Discrete}
        """Make observation spaces based on the communication spaces of the agents.

        This method creates a dictionary of observation spaces for higher-level agents based on
        the communication spaces of their connected lower-level agents defined in uplinks.

        Returns:
            GymDict: A dictionary containing observation spaces for each higher-level agent.
                    The structure is {hl_agent: {agent_name: communication_space}}.
                    If no uplinks are defined, returns an empty GymDict.

        Raises:
            ValueError: If any agent in uplinks doesn't have a defined communication_space attribute.

        Notes:
            - Requires self.uplinks to be defined as a dictionary mapping higher-level agents
              to lists of lower-level agent names.
            - Each agent must have a communication_space attribute defining its observation space.
            - If self.verbose >= 2, prints the observation space structure.
        """
        observation_space = {}
        if self.uplinks is None:
            return GymDict(observation_space)
        for hl_agent, agent_names in self.uplinks.items():
            agent_spaces = {}
            for agent_name in agent_names:
                if not hasattr(self.agents[agent_name], "communication_space"):
                    raise ValueError(
                        f"Agent {agent_name} does not have an communication_space"
                    )
                agent_spaces.update(self.agents[agent_name].communication_space)  # type: ignore
            observation_space[hl_agent] = GymDict(agent_spaces)

        if self.verbose >= 2:
            print(f"{self.name} -  observation space")
            print(f"\t {observation_space}")
        return GymDict(observation_space)

    def _make_action_space(self) -> GymDict:  # {str: Box | Discrete}
        """Make action space based on the observation spaces of the agents.

        This method creates an action space dictionary where each high-level agent's action space
        is defined by the observation spaces of its associated lower-level agents.

        Returns:
            GymDict: A dictionary mapping high-level agent names to their action spaces.
                Each action space is a GymDict containing the observation spaces of the
                controlled lower-level agents. If no uplinks are defined, returns an empty GymDict.

        Raises:
            ValueError: If any of the controlled agents does not have an observation_space attribute.

        Note:
            The action space of each high-level agent is constructed by combining the
            observation spaces of all lower-level agents it controls, as defined in self.uplinks.
        """
        action_space = {}
        if self.uplinks is None:
            return GymDict(action_space)
        for hl_agent, agent_names in self.uplinks.items():
            agent_spaces = {}
            for agent_name in agent_names:
                if not hasattr(self.agents[agent_name], "observation_space"):
                    raise ValueError(
                        f"Agent {agent_name} does not have an observation_space"
                    )
                agent_spaces.update(self.agents[agent_name].observation_space)
            action_space[hl_agent] = GymDict(agent_spaces)
        if self.verbose >= 2:
            print(f"{self.name} -  action space")
            print(f"\t {action_space}")
        return GymDict(action_space)

    def seed(self, seed):
        """
        Sets the random seed for this environment and its components.

        This method initializes random number generators for the environment by setting
        the seed for Python's random module, NumPy's random number generator, all agents
        in the environment, and the underlying environment if it supports seeding.

        Args:
            seed (int): The seed value to use for random number generation.

        Returns:
            None
        """
        random.seed(seed)
        np.random.seed(seed)
        for agent in self.agents.values():
            agent.seed(seed=seed)

        if hasmethod(self.env, "seed"):
            self.env.seed(seed)

    def set_logger(self, logger: SummaryWriter | None, save_path: str | Path):
        """Sets logger and save path for the environment.

        This method initializes or updates the logging configuration for the environment by setting
        the logger instance and creating the necessary directory structure for saving data.

        Args:
            logger (SummaryWriter | None): The logger instance to be used for logging. Can be None.
            save_path (str | Path): The base path where environment data will be saved. Will be
                combined with the environment name to create the final save location.

        Notes:
            - Creates the save directory if it doesn't exist
            - Stores the logger instance as self.logger
            - Constructs the final save path by joining save_path with environment name
            - Prints status messages if verbose level is >= 2
        """
        self.save_path = Path(save_path) / self.name
        if self.verbose >= 2:
            print(f"{self.name} -  Updating logger and save path: {self.save_path}")

        if not self.save_path.exists():
            if self.verbose >= 2:
                print(
                    f"{self.name} -  Save path {self.save_path} does not exists. Creating it."
                )

            os.makedirs(self.save_path, exist_ok=True)
        self.logger = logger

    def reset(self) -> Tuple[Dict[str, Dict[str, np.ndarray]] | None, dict | None]:
        """
        This method performs the following operations:
        1. Logs the episode return if logging is enabled
        2. Saves and empties the trace if save_path is specified
        3. Resets all agents that have a reset method
        4. Resets the lower level environment
        5. Generates observation and info messages
        6. Resets episode-related counters

        Each returned element is a dict where the keys are the names of the higher level agents.

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], dict]: A tuple containing:
                - msg_obs: Dictionary of observations for each agent
                - msg_infos: Dictionary of info messages for each agent
        """
        if self.verbose >= 1:
            print(f"{self.name} -  Reset")

        # Log episode return
        if self.episode_idx >= 0:
            if self.logger is not None:
                self.logger.add_scalar(
                    f"{self.name}/total_return",
                    np.sum([ret for ret in self.episode_returns.values()]),
                    self.level_ts,
                )
            if self.verbose >= 2:
                print(
                    f"{self.name} -  level_step: {self.level_ts} - Ep. return={np.sum([ret for ret in self.episode_returns.values()])}"
                )

        # Save trace and empty it
        if self.save_path is not None and self.trace is not None and len(self.trace):
            self.trace.save_trace(save_path=self.save_path, episode=self.episode_idx)
            self.trace.empty()

        # Reset agents
        for agent in self.agents.values():
            if hasmethod(agent, "reset"):
                agent.reset()  # type: ignore

        # Reset lower level env
        env_obs, env_inf = self.env.reset()
        env_obs = self.save_obs_from_bottom(env_obs=env_obs)

        msg_obs, _, _, _, msg_infos = self.make_message(
            env_obs=env_obs,
            env_rew=None,
            env_term=None,
            env_trunc=None,
            env_inf=env_inf,
        )

        self.episode_returns = {ag: 0.0 for ag in self.agents}
        self.episode_ts = 0
        self.episode_idx += 1
        self._decision_obs = {}
        self._interval_rew = {ag: 0.0 for ag in self.agents}
        self._interval_done = {ag: False for ag in self.agents}
        self._interval_stored = {ag: True for ag in self.agents}
        return msg_obs, msg_infos

    def save_obs_from_bottom(
        self, env_obs: Dict[str, np.ndarray] | Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Wraps and stores the observation from the bottom level of the hierarchy.

        This method takes observations from a lower level and reformats them if necessary to maintain
        a consistent hierarchical structure. It supports both nested and flat observation dictionaries.

        Args:
            env_obs (Dict[str, np.ndarray] | Dict[str, Dict[str, np.ndarray]]): Observations from the lower level.
                Can be either a flat dictionary mapping agent IDs to observations,
                or a nested dictionary with agent-subagent structure.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: A nested dictionary where:
                - First level keys are agent IDs
                - Second level keys are subagent IDs
                - Values are the corresponding observations as numpy arrays

        Note:
            The method stores the processed observations in self.obs_from_bottom before returning them.
        """
        if is_nested_dict(env_obs):
            self.obs_from_bottom = env_obs
            return env_obs
        else:
            reformatted_obs = {}
            for agent, subagents in self.downlinks.items():
                reformatted_obs[agent] = {subag: env_obs[subag] for subag in subagents}
            self.obs_from_bottom = reformatted_obs
            return reformatted_obs

    def make_agent_obs(
        self,
        action_from_top: Dict[str, np.ndarray] | None,
        obs_from_bottom: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generates the observation for the agents on this level by using the observation from the bottom and
        the actions from top

        This function combines or replaces observations from lower level agents with actions from higher level
        agents to create the observation space for the current level's agents.

        Args:
            action_from_top (Dict[str, np.ndarray] | None): Dictionary mapping agent names to their actions from
                the higher level, or None if this is the top level
            obs_from_bottom (Dict[str, Dict[str, np.ndarray]]): Dictionary of dictionaries containing observations
                from lower level agents. First key is current level agent name, second key is lower level agent name

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of dictionaries containing the combined/replaced observations.
                First key is current level agent name, second key is lower level agent name
        """
        if action_from_top is None:
            return obs_from_bottom

        agent_obs = {}  # It will have shape {this_level_agent: {bottom_agent: obs}}
        if self.concat_obs:
            for agent_name in action_from_top:  # These are the agents in this level
                obs = {  # This combines the observation from bottom with the action from top
                    sub_name: np.concat(
                        [
                            obs_from_bottom[agent_name][sub_name],
                            np.atleast_1d(action_from_top[agent_name]),
                        ]
                    )
                    for sub_name in obs_from_bottom[agent_name]
                }
                agent_obs[agent_name] = obs
        else:
            for agent_name in action_from_top:
                obs = {  # This replaces the observation from bottom with the action from top
                    sub_name: np.atleast_1d(action_from_top[agent_name])
                    for sub_name in obs_from_bottom[agent_name]
                }
                agent_obs[agent_name] = obs
        return agent_obs

    def step(
        self,
        action: Dict[str, np.ndarray] | None,
        training: bool = True,
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]] | None,
        Dict[str, float] | None,
        Dict[str, bool] | None,
        Dict[str, bool] | None,
        dict | None,
    ]:
        """Perform a step in the hierarchical environment, processing actions from top to bottom and then bottom to top.

        Args:
            action (Dict[str, np.ndarray] | None): Actions from the higher level, containing observations for each agent at this level.
                None if there are no actions from higher level.
            training (bool, optional): Flag indicating whether the environment is in training mode. Defaults to True.

        Returns:
            Tuple containing five dictionaries:
                Dict[str, Dict[str, np.ndarray]]: Observations for each agent
                Dict[str, float]: Rewards for each agent
                Dict[str, bool]: Termination flags for each agent
                Dict[str, bool]: Truncation flags for each agent
                dict: Additional information dictionary

        Flow:
            1. Creates agent observations using high-level actions and bottom-level observations
            2. Selects new actions every action_ifreq lower-level steps, holding them otherwise
            3. Steps through the environment with agent actions
            4. Accumulates rewards over the decision interval and, in training mode, stores one
               transition per decision (semi-MDP), plus a final observation at episode end
            5. Updates traces and episode returns
            6. Periodically saves agent models if save_path is specified

        Note:
            - The method handles both the downward (top to bottom) and upward (bottom to top) flow of information
            - Training data is stored only when training=True
            - Final observations are stored when episodes terminate"""

        # Act
        # ---------------------------
        agent_obs = self.make_agent_obs(
            action_from_top=action, obs_from_bottom=self.obs_from_bottom
        )

        # Act only every action_ifreq steps of the lower level. Between decisions the
        # action is held, and the level accumulates the rewards of the interval so that
        # a single transition (decision obs, action, summed reward) is stored per decision.
        # TODO the frequency should be decided by the agent not by the env
        if self.episode_ts % self.action_ifreq == 0:
            # Pass the higher level actions to the agents of this level as obs
            self.env_actions = self.act(observation=agent_obs, training=training)
            # Open a new decision interval
            self._decision_obs = agent_obs
            self._interval_rew = {ag: 0.0 for ag in self.agents}
            self._interval_done = {ag: False for ag in self.agents}
            self._interval_stored = {ag: False for ag in self.agents}

        # Pass the actions of the agents of this level to the lower level env and get rew and ll_obs
        if isinstance(self.env, LevelEnv):
            env_obs, env_rew, env_term, env_trunc, env_inf = self.env.step(
                self.env_actions, training=training
            )
            assert (
                env_obs is not None
            ), f"The level {self.env.name} should return to {self.name} valid observations"
            assert (
                env_term is not None
            ), f"The level {self.env.name} should return to {self.name} valid terminations"
            assert (
                env_trunc is not None
            ), f"The level {self.env.name} should return to {self.name} valid truncations"
            assert (
                env_rew is not None
            ), f"The level {self.env.name} should return to {self.name} valid rewards"
        else:
            env_obs, env_rew, env_term, env_trunc, env_inf = self.env.step(
                self.env_actions
            )
        env_obs = self.save_obs_from_bottom(env_obs=env_obs)

        dones = {agent: (env_term[agent] or env_trunc[agent]) for agent in self.agents}
        # ---------------------------

        # Make message
        # ---------------------------
        msg_obs, msg_rew, msg_term, msg_trunc, msg_infos = self.make_message(
            env_obs=env_obs,
            env_rew=env_rew,
            env_trunc=env_trunc,
            env_term=env_term,
            env_inf=env_inf,
        )
        # ---------------------------

        # Accumulate the interval reward and termination for every agent of this level
        # ---------------------------
        for agent_name in self.agents:
            self._interval_rew[agent_name] += env_rew[agent_name]
            self._interval_done[agent_name] = (
                self._interval_done[agent_name] or dones[agent_name]
            )
        # The interval closes when the next decision is due, or when the episode ends
        interval_complete = (self.episode_ts + 1) % self.action_ifreq == 0
        # ---------------------------

        # Train
        # ---------------------------
        last_agent_obs = None
        if training:
            # No need for this when not training. (For now :P)
            for agent_name, agent in self.agents.items():
                # Get the actions from this agent
                agent_actions = {
                    subagent: self.env_actions[subagent]
                    for subagent in self.downlinks[agent_name]
                }

                # One transition per decision: the observation at decision time, the action
                # taken, and the (undiscounted) sum of the rewards received over the interval.
                # With action_ifreq == 1 this is the usual one transition per step.
                if (interval_complete or dones[agent_name]) and not self._interval_stored[
                    agent_name
                ]:
                    agent.store(
                        state=self._decision_obs[agent_name],
                        action=agent_actions,
                        reward=self._interval_rew[agent_name],
                        done=self._interval_done[agent_name],
                    )
                    self._interval_stored[agent_name] = True

                # NOTE
                # We create a new agent_obs with the last state from the env and the same top action!
                # we do this cause there will be no new action coming from top, as the top level will call the reset
                # And then we store the last transition!
                if dones[agent_name]:
                    if last_agent_obs is None:
                        last_agent_obs = self.make_agent_obs(
                            action_from_top=action, obs_from_bottom=env_obs
                        )

                    agent.store(
                        state=last_agent_obs[agent_name],
                        action=None,
                        reward=None,
                        done=dones[agent_name],
                    )

            self.update_step()
        # ---------------------------

        # Save important infos
        # ---------------------------
        if self.trace is not None:
            self.trace.add(
                episode=self.episode_idx,
                actions=self.env_actions,
                observations=agent_obs,
                rewards=env_rew,
                terminations=env_term,
                truncations=env_trunc,
                # These are the ones sent to the higher level
                return_observations=msg_obs,
                return_rewards=msg_rew,
            )

            if last_agent_obs is not None:
                self.trace.add_final_obs(last_agent_obs)

        for agent in self.episode_returns:
            self.episode_returns[agent] += env_rew[agent]

        if self.level_ts % 100000 == 0 and self.save_path is not None:
            if self.verbose >= 2:
                print(
                    f"{self.name} -  Saving model at level step {self.level_ts} in {self.save_path}"
                )
            with open(self.save_path / "last_model.json", "w") as f:
                json.dump({"save_step": self.level_ts}, f)
            for agent_name, agent in self.agents.items():
                agent.save_agent(save_path=self.save_path, name=agent_name)
        # ---------------------------

        self.level_ts += 1
        self.episode_ts += 1
        return msg_obs, msg_rew, msg_term, msg_trunc, msg_infos

    def update_step(self):
        """
        This function is called for each step and iterates through all agents in the level,
        calling their individual update_step methods. The level's global timestep and logger
        are passed to each agent's update step.

        This function is called for each step.
        """
        for agent in self.agents.values():
            agent.update_step(global_step=self.level_ts, writer=self.logger)

    def save(self, save_path: str | Path):
        """Save the level and its agents to disk.

        This method saves all agents in the level to the specified directory. The directory
        will be created if it doesn't exist. Each agent is saved with its corresponding name
        as a prefix.

        Args:
            save_path (str | Path): Base directory where to save the level data. A subdirectory
                                   with the level's name will be created inside this path.

        Example:
            >>> level.save("/path/to/save/directory")
            # Creates /path/to/save/directory/level_name/ and saves agents there

        Notes:
            - If verbose >= 1, prints information about the saving process
            - If verbose >= 2, also prints if the save path differs from the original one
        """
        save_path = Path(save_path) / self.name
        if not save_path.exists():
            os.makedirs(save_path, exist_ok=True)

        if self.verbose >= 1:
            print_message = f"{self.name} -  Save function called with {save_path}. "
            if self.verbose >= 2 and save_path != self.save_path:
                print_message += f"Original save path was different: {self.save_path}. "
            print_message += "Saving agents."
            print(print_message)

        for agent_name, agent in self.agents.items():
            agent.save_agent(save_path=save_path, name=f"{agent_name}")

    def load(self, load_path: str | Path) -> bool:
        """Loads all the agents of the level from the specified path.

        Args:
            load_path (str | Path): The path from where to load the agents.

        Returns:
            bool: True if all agents were loaded successfully, False if any agent failed to load.

        Example:
            >>> level.load("/path/to/agents")
            True
        """
        all_loaded = True
        for agent_name, agent in self.agents.items():
            loaded = agent.load_agent(load_path=load_path, name=agent_name)
            if not loaded:
                print(f"Model {agent_name} cannot be loaded")
            all_loaded = all_loaded and loaded
        return all_loaded

    def make_message(
        self,
        env_obs: Dict[str, Dict[str, np.ndarray]],
        env_rew: Dict[str, float] | None,
        env_term: Dict[str, bool] | None,
        env_trunc: Dict[str, bool] | None,
        env_inf: dict | None,
    ) -> (
        Tuple[
            Dict[str, Dict[str, np.ndarray]],
            Dict[str, float],
            Dict[str, bool],
            Dict[str, bool],
            dict,
        ]
        | Tuple[None, None, None, None, None]
    ):
        """
        This is the bottom->top direction in the hierarchy where each level communicates its status
        to the level above it. The message contains observations, rewards, termination flags,
        truncation flags and info dictionaries for each higher level agent.

        Args:
            env_obs (Dict[str, Dict[str, np.ndarray]]): Observations from lower level, structured as
                {agent_name: {observing_agent: observation_array}}
            env_rew (Dict[str, float] | None): Rewards from lower level for each agent
            env_term (Dict[str, bool] | None): Termination flags from lower level for each agent
            env_trunc (Dict[str, bool] | None): Truncation flags from lower level for each agent
            env_inf (dict | None): Additional info from lower level for each agent

        Returns:
            Tuple containing 5 dictionaries, each having higher level agent names as keys:
                - observations (Dict[str, Dict[str, np.ndarray]]): Processed observations for each higher agent
                - rewards (Dict[str, float]): Sum of rewards from lower level agents
                - terminations (Dict[str, bool]): OR of termination flags from lower agents
                - truncations (Dict[str, bool]): OR of truncation flags from lower agents
                - infos (dict): Collected info dicts from lower agents

        Notes:
            If self.uplinks is None, returns the input arguments unchanged.
            For rewards, each higher level agent receives the sum of rewards from its assigned lower level agents. NOTE: This will change in future releases, so the reward function can be customized
            For terminations/truncations, flags are OR-ed across assigned lower level agents.
        """
        # Generate this level observation etc
        if self.uplinks is None:
            return None, env_rew, env_term, env_trunc, env_inf  # type: ignore

        observations = {hl_agent: {} for hl_agent in self.uplinks}
        rewards = {hl_agent: 0.0 for hl_agent in self.uplinks}
        terminations = {hl_agent: False for hl_agent in self.uplinks}
        truncations = {hl_agent: False for hl_agent in self.uplinks}
        infos = {hl_agent: {} for hl_agent in self.uplinks}
        for hl_agent, agent_names in self.uplinks.items():
            for agent in agent_names:
                observations[hl_agent][agent] = self.agents[agent].comm(env_obs[agent])

                # TODO add some form of level dependent reward
                # TODO the hl agent receive the sum of reward for the level agents assigned to him
                # TODO check maybe it's better to send them separatedly?
                if env_rew is not None:
                    rewards[hl_agent] += env_rew[agent]

                if env_term is not None:
                    terminations[hl_agent] = terminations[hl_agent] or env_term[agent]
                if env_trunc is not None:
                    truncations[hl_agent] = truncations[hl_agent] or env_trunc[agent]
                if env_inf is not None:
                    infos[hl_agent][agent] = env_inf[agent]
                else:
                    infos[hl_agent][agent] = {}
        return observations, rewards, terminations, truncations, infos

    def act(
        self, observation: Dict[str, Dict[str, Any]], training: bool = False
    ) -> Dict[str, Any]:
        """
        Generates this level actions for the lower level.
        This is the top->bottom direction in the hierarchy.

        Args:
            observation (Dict[str, Dict[str, Any]]): Nested dictionary containing observations.
                First level key is this level's agent name,
                Second level key is bottom level agent name,
                Value is the corresponding observation.
            training (bool, optional): Whether the action is generated during training.
                Defaults to False.

        Returns:
            Dict[str, Any]: Actions dictionary from this level for the lower one.
                Keys are agent names and values are their corresponding actions.

        Example:
            observation = {
                'agent1': {'bottom_agent1': obs1, 'bottom_agent2': obs2},
                'agent2': {'bottom_agent3': obs3}
            }
        """
        agent_actions = {}
        for agent_name in observation:
            if training:
                agent_act = self.agents[agent_name].act_train(
                    observation[agent_name],
                    global_step=self.level_ts,
                )
            else:
                agent_act = self.agents[agent_name].act(observation[agent_name])
            agent_actions.update(agent_act)
        return agent_actions
