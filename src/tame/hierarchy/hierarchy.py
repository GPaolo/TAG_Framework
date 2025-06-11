from typing import Dict, List, Any
from tame.hierarchy.level_env import LevelEnv
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium.spaces import Discrete, Box, Dict as GymDict
from dataclasses import dataclass, field
from tame.hierarchy.base_agent import LevelAgent
from typing import Type
from pettingzoo import ParallelEnv
import torch
from tame.utils.space_utils import merge_spaces, extend_box_space, extend_dict_box_space


@dataclass
class AgentConfig:
    """
    AgentConfig is a configuration class for defining the properties of an agent.

    Attributes:
        name (str): The name of the agent.
        communication_space (GymDict | None): The communication space for the agent, which can be a GymDict or None.
        agent_class (Type[LevelAgent]): The class of the agent.
        agent_kwargs (dict): A dictionary of keyword arguments for the agent. Defaults to an empty dictionary.
        device (torch.device): The device on which the agent will run. Defaults to a CPU device.
    """

    name: str
    communication_space: GymDict | None
    agent_class: Type[LevelAgent]
    agent_kwargs: dict = field(default_factory=dict)
    device: torch.device = torch.device("cpu")


@dataclass
class LevelConfig:
    """
    LevelConfig is a data class that defines the configuration for a hierarchical level in a multi-agent environment.

    Attributes:
        name (str): The name of the level.
        agents (List[AgentConfig]): A list of agent configurations for the level.
        env (ParallelEnv | LevelEnv): The environment associated with the level, which can be either a parallel environment or a level environment.
        uplinks (Dict[str, List[str]] | None): A dictionary mapping higher-level agents to lists of lower-level agents. This can be None if there are no uplinks.
        downlinks (Dict[str, List[str]]): A dictionary mapping agents to lists of environment agents.
        action_frequency (int): The frequency of actions taken by the agents. Default is 1.
        trace_type (str): The type of trace to be used, default is "reward".
        concat_obs (bool): A flag indicating whether to concatenate observations. Default is False.
        action_space (GymDict | None): The action space for the agents, which can be None.
    """

    name: str
    agents: List[AgentConfig]
    env: ParallelEnv | LevelEnv
    uplinks: Dict[str, List[str]] | None  # {higher_agent: [lower_agents]}
    downlinks: Dict[str, List[str]]  # {agent: [env_agents]}
    action_frequency: int = 1
    trace_type: str = "reward"
    concat_obs: bool = False
    action_space: GymDict | None = None


class Hierarchy:
    """A class to represent a hierarchical structure of levels and agents.

    Attributes:
        levels (List[LevelEnv]): A list of levels in the hierarchy.
        env (Any): The environment connected to the hierarchy.

    Methods:
        interface_level:
            Returns the top level of the hierarchy.

        set_logger(logger: SummaryWriter | None, save_path: str | Path) -> None:
            Sets the logger for all levels in the hierarchy.

        add_level(level: LevelEnv) -> None:
            Adds a level to the hierarchy.

        add_level_config(level_cfg: LevelConfig) -> None:
            Adds a level to the hierarchy based on the provided configuration.

        connect(env: Any) -> None:
            Connects the hierarchy to the given environment.

        reset() -> Tuple[Dict[str, Any], Dict[str, Any]]:
            Resets the entire hierarchy and returns the initial observations and info.

        step(action: Dict[str, Any] | None) -> Tuple[Dict[str, Any], Dict[str, Any], bool, bool, Dict[str, Any]]:
            Steps through the hierarchy from top to bottom with the given action.

        save(path: Path) -> None:
            Saves all levels in the hierarchy to the specified path.

        load(path: Path) -> bool:
            Loads all levels in the hierarchy from the specified path.

        act(observation: Dict[str, Any]) -> Dict[str, Any]:
            Processes observations from the environment through the hierarchy and returns actions.

        tree() -> dict:
            Constructs and returns the hierarchical tree structure.

        print_tree(tree: None | dict = None, indent: int = 0, prefix: str = "") -> None:
            Prints the hierarchical tree structure.

        print_hierarchy_details() -> None:
            Prints detailed information about the hierarchy structure, including level and agent spaces."""

    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        self.levels: List[LevelEnv] = []
        self.env = None

    @property
    def interface_level(self):
        """Returns the top level of the hierarchy.

        Returns:
            LevelEnv: The top level of the hierarchy.
        """
        return self.levels[0]

    def set_logger(self, logger: SummaryWriter | None, save_path: str | Path) -> None:
        """
        Set logger for all levels.

        Args:
            logger (SummaryWriter | None): The logger instance to be used. If None, logging is disabled.
            save_path (str | Path): The path where logs should be saved.

        Returns:
            None
        """
        for level in self.levels:
            level.set_logger(logger=logger, save_path=save_path)

    def add_level(self, level: LevelEnv) -> None:
        """
        Add a level to the hierarchy.

        Args:
            level (LevelEnv): The level to be added to the hierarchy.
        """
        self.levels.append(level)

    def add_level_config(self, level_cfg: LevelConfig) -> None:
        """Adds a level to the hierarchy based on the provided configuration.

        This method creates a new level in the hierarchy by instantiating agents according to the level
        configuration and connecting them with the previous level. It handles observation space modifications
        when concatenating observations from higher levels.

        Args:
            level_cfg (LevelConfig): Configuration object containing all necessary parameters to create
                a new level, including:
                - env: Environment for this level
                - agents: List of agent configurations
                - name: Level name
                - concat_obs: Whether to concatenate observations from higher levels
                - action_space: Space of actions from higher levels (required if concat_obs=True)
                - action_frequency: Frequency of actions
                - trace_type: Type of trace to use
                - uplinks/downlinks: Communication links between levels

        Raises:
            ValueError: If concat_obs is True but no action_space is provided
            TypeError: If observation spaces are incompatible when merging spaces

        Notes:
            - If concat_obs is True, the observation space of each agent is extended to include
              actions from higher levels
            - The first level added becomes the base environment of the hierarchy
        """
        prev_level = level_cfg.env

        # Create agents
        agents = {}
        for cfg in level_cfg.agents:
            # Find action space from top
            # ---------------------------
            agent_additional_obs_space = None
            if level_cfg.concat_obs:
                if level_cfg.action_space is None:
                    raise ValueError(
                        f"Must provide an action space for the level {level_cfg.name} if concat_obs is True"
                    )

                for upper_agent in level_cfg.action_space:
                    if cfg.name in level_cfg.action_space[upper_agent].keys():  # type: ignore
                        agent_additional_obs_space = level_cfg.action_space[
                            upper_agent
                        ][cfg.name]  # type: ignore
                        break
            # ---------------------------

            # Make new observation space for agent
            # We take observation space of the env and add the action size from the top
            # ---------------------------
            # TODO make sure that we can have different names for the agents in the bottom level
            # TODO so, we should handle the fact that the observation space from the env is not a nested dict!
            # TODO so something that could be done is to have: if prev_lev not LevelEnv: wrap spaces...
            if isinstance(agent_additional_obs_space, Discrete):
                if isinstance(prev_level.observation_space(cfg.name), Box):
                    agent_obs_space = extend_box_space(
                        original_space=prev_level.observation_space(cfg.name),  # type: ignore
                        add_size=1,
                    )
                elif isinstance(prev_level.observation_space(cfg.name), GymDict):
                    agent_obs_space = extend_dict_box_space(
                        original_space=prev_level.observation_space(cfg.name),  # type: ignore
                        add_size=1,
                    )
                else:
                    raise TypeError(
                        f"For level {level_cfg.name} and agent {cfg.name} previous level observation space must be Box or GymDict[str: Box]."
                    )
            elif isinstance(agent_additional_obs_space, Box):
                if isinstance(prev_level.observation_space(cfg.name), Box):
                    agent_obs_space = merge_spaces(
                        [
                            prev_level.observation_space(cfg.name),  # type: ignore
                            agent_additional_obs_space,
                        ]
                    )
                else:
                    raise TypeError(
                        f"For level {level_cfg.name} and agent {cfg.name} both observation spaces must be of type Box to merge."
                    )
            else:
                agent_obs_space = prev_level.observation_space(cfg.name)

            if not isinstance(agent_obs_space, GymDict):
                agent_obs_space = GymDict({cfg.name: agent_obs_space})

            agent_action_space = prev_level.action_space(cfg.name)
            if not isinstance(agent_action_space, GymDict):
                agent_action_space = GymDict({cfg.name: agent_action_space})
            # ---------------------------

            # Instantiate agent
            agents[cfg.name] = cfg.agent_class(
                name=cfg.name,
                observation_space=agent_obs_space,
                action_space=agent_action_space,
                communication_space=cfg.communication_space,
                device=cfg.device,
                args=cfg.agent_kwargs.get("args", None),
            )

        # Create level
        level = LevelEnv(
            agents=agents,
            uplinks=level_cfg.uplinks,
            downlinks=level_cfg.downlinks,
            env=prev_level,
            name=level_cfg.name,
            action_ifreq=level_cfg.action_frequency,
            trace_type=level_cfg.trace_type,
            concat_obs=level_cfg.concat_obs,
            action_space=level_cfg.action_space,
        )

        if len(self.levels) == 0:
            self.env = level_cfg.env
        self.add_level(level)

    def connect(self, env: Any) -> None:
        """Connect hierarchy to environment.

        This method establishes the connection between the hierarchy and the target environment
        by setting the environment reference for both the hierarchy instance and its base level.

        Args:
            env (Any): The environment to connect to. Can be any type implementing the required interface.

        Returns:
            None
        """
        self.env = env
        self.levels[0].env = env

    def reset(self):
        """Reset entire hierarchy.

        This method resets the entire hierarchical structure by calling the reset method of the
        topmost level in the hierarchy.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observation after reset
                - info: A dictionary containing auxiliary information
        """
        obs, info = self.levels[-1].reset()
        return obs, info

    def step(self, action: Dict[str, Any] | None):
        """Step through hierarchy from top to bottom.

        This method executes a single step through the hierarchy, starting from the top level
        and propagating down to the bottom level.

        Args:
            action (Dict[str, Any] | None): Action dictionary to be executed. Can be None if no action is required.

        Returns:
            The result of stepping through the bottom-most level of the hierarchy.
        """
        return self.levels[-1].step(action)

    def save(self, path: Path):
        """Save the hierarchy by saving all levels to the specified path.

        Args:
            path (Path): The directory path where all levels will be saved.

        Example:
            hierarchy = Hierarchy()
            hierarchy.save(Path("/path/to/save"))
        """
        for level in self.levels:
            level.save(path)

    def load(self, path: Path) -> bool:
        """Loads all levels in the hierarchy from the specified path.

        Args:
            path (Path): Base directory path where the levels should be loaded from.
                         Each level will be loaded from a subdirectory matching its name.

        Returns:
            bool: True if all levels were loaded successfully, False if any level failed to load.

        Example:
            >>> hierarchy = Hierarchy()
            >>> hierarchy.load(Path("/path/to/levels"))
            True
        """
        return all(level.load(path / level.name) for level in self.levels)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow observations from environment to travel up the hierarchy and actions to travel down.

        This method implements a two-pass hierarchical process:
        1. Bottom-up pass: Observations travel up the hierarchy, with each level generating messages
            for the level above it.
        2. Top-down pass: Actions are generated at each level and passed down the hierarchy.

        Args:
             observation (Dict[str, Any]): Environment observation dictionary mapping agent names to
                  their observations.

        Returns:
             Dict[str, Any]: Dictionary of actions for the environment mapping agent names to their
                  corresponding actions.

        Flow:
             1. Bottom-up:
                  - Each level processes observations from below
                  - Generates messages to pass upwards
                  - Messages become observations for next level
             2. Top-down:
                  - Starting from top level
                  - Each level receives action from above
                  - Combines with bottom-up message to generate new action
                  - Passes action down to next level
        """
        # First propagate observations up the hierarchy from bottom to top
        obs = observation

        # Bottom-up pass: Collect messages from each level
        for level in self.levels:
            # Save and format observation for this level
            prepared_obs = level.save_obs_from_bottom(obs) # type: ignore

            # Generate message to send up
            msg = level.make_message(
                env_obs=prepared_obs,
                env_rew=None,
                env_term=None,
                env_trunc=None,
                env_inf=None,
            )[0]  # Take first element since make_message returns tuple

            obs = msg  # Pass message as observation to next level

        # Top-down pass: Generate actions at each level
        action = None

        # Iterate through levels in reverse, from top to bottom
        for i in range(len(self.levels) - 1, -1, -1):
            level = self.levels[i]

            # Create observation for this level by combining bottom-up message
            # with top-down action
            level_obs = level.make_agent_obs(
                action_from_top=action, obs_from_bottom=level.obs_from_bottom
            )

            # Get action from this level to pass down
            action = level.act(observation=level_obs)

        return action  # type: ignore

    def tree(self):
        """Returns a hierarchical tree representation of the associations between levels.

        This method constructs a nested dictionary representing the hierarchical structure,
        starting from the lowest level and building up through the uplinks between levels.
        Each level's associations are merged into a tree structure where keys are nodes
        and values are their children/descendants.

        Returns:
            dict: A nested dictionary representing the hierarchical tree structure.
                  The keys are parent nodes and values are dictionaries containing
                  their children and further descendants.

        Example:
            If Level 0 has nodes [1,2,3] and Level 1 has nodes [A,B] with
            A linked to [1,2] and B linked to [3], the result would be:
            {'A': {'1': {}, '2': {}}, 'B': {'3': {}}}
        """
        tree = self.levels[0].downlinks
        for level in self.levels:
            if level.uplinks is None:
                break

            local_tree = {}
            for up, down in level.uplinks.items():
                local_tree[up] = {d: tree[d] for d in down}
            tree = local_tree
        return tree

    def print_tree(self, tree: None | dict = None, indent: int = 0, prefix: str = ""):
        """
        Prints a hierarchical tree representation of the agent hierarchy.

        This method visualizes the hierarchical structure using ASCII characters, where leaf nodes
        (environments) are marked with '(env)' and intermediate nodes represent higher-level agents.

        Args:
            tree (dict | None, optional): The hierarchical tree structure to print. If None,
                uses the tree from self.tree(). Defaults to None.
            indent (int, optional): The current indentation level. Used for recursive calls.
                Defaults to 0.
            prefix (str, optional): The prefix string for the current line. Used for recursive
                calls to maintain the tree structure. Defaults to "".

        Example:
            For a hierarchy with two high-level agents and their environments:
            Agent1
            ├── env1 (env)
            └── env2 (env)
            Agent2
            └── env3 (env)
        """
        if tree is None:
            tree = self.tree()

        items = list(tree.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            # Higher level agents (with children) don't get env flag
            print(prefix + current_prefix + str(key))

            new_prefix = prefix + ("    " if is_last else "│   ")
            if isinstance(value, dict):
                self.print_tree(value, indent + 1, new_prefix)
            elif isinstance(value, list):
                for j, item in enumerate(value):
                    is_last_item = j == len(value) - 1
                    item_prefix = "└── " if is_last_item else "├── "
                    # Add (env) flag to leaf nodes
                    print(new_prefix + item_prefix + str(item) + " (env)")

    def print_hierarchy_details(self):
        """Print detailed information about hierarchy structure, including level and agent spaces.

        This method provides a comprehensive visualization of the hierarchy's structure, printing
        information about each level including:
        - Level name and index
        - Level-wide observation and action spaces
        - Uplink and downlink connectivity between agents
        - Per-agent details:
            - Observation space
            - Action space
            - Communication space
            - Input connections from lower-level agents (for non-bottom levels)

        The output is formatted with clear section headers and separators for readability.

        Returns:
            None
        """
        print("\n=== Hierarchy Details ===\n")

        for level_idx, level in enumerate(self.levels):
            print(f"Level {level_idx}: {level.name}")
            print("─" * 60)

            # Print level spaces
            print("\nLevel Spaces:")
            print(f"  Observation Space: {level.observation_spaces}")
            print(f"  Action Space: {level.action_spaces}")

            # Print connectivity
            if level.uplinks:
                print("\nUplinks:")
                for up, down in level.uplinks.items():
                    print(f"  {up} → {down}")

            if level.downlinks:
                print("\nDownlinks:")
                for agent, children in level.downlinks.items():
                    print(f"  {agent} → {children}")

            # Print agent details
            print("\nAgents:")
            for agent_name, agent in level.agents.items():
                print(f"  {agent_name}:")
                print(f"    Observation space: {agent.observation_space}")
                print(f"    Action space: {agent.action_space}")
                print(f"    Communication space: {agent.communication_space}")

                # Show space transformations if applicable
                if level_idx > 0:  # Not bottom level
                    print("    Receives from below:")
                    for child in level.downlinks.get(agent_name, []):
                        child_agent = self.levels[level_idx - 1].agents[child]
                        print(f"      {child}: {child_agent.communication_space}")

            print("\n" + "=" * 60 + "\n")
