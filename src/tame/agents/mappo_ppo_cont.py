import os
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from pettingzoo import ParallelEnv
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from tame.agents.monolithic_ppo import Agent as PPO
from tame.agents.monolithic_ppo import Args as PPOArgs
from tame.agents.mappo import Agent as MAPPO
from tame.agents.mappo import Args as MAPPOArgs
from typing import Dict
from tame.hierarchy.base_agent import BaseAgent
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Box
from tame.utils.utils import filter_unexpected_fields
from tame.utils.config import ArgsInterface
from tame.hierarchy.hierarchy import Hierarchy, LevelConfig, AgentConfig
import json


@filter_unexpected_fields
@dataclass_json
@dataclass
class Args(ArgsInterface):
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int | None = 1
    torch_deterministic: bool = True
    verbose: bool = False
    cuda: int = 0
    save_model: bool = True
    save_all_trace: bool = True
    total_timesteps: int = 500000
    episode_length: int = 100
    ifreq_bottom: int = 1
    ifreq_mid: int = 1
    ifreq_top: int = 1
    mappo_args: MAPPOArgs = field(default_factory=MAPPOArgs)
    ppo_args: PPOArgs = field(default_factory=PPOArgs)

    def __post_init__(self):
        self.subargs = ["mappo_args", "ppo_args"]
        super().__post_init__()

        self.ppo_args.num_minibatches = 8
        self.ppo_args.ent_coef = 0.0
        self.ppo_args.learning_rate = 0.001

        self.mappo_args.batch_size = 10000
        self.mappo_args.lr = 0.001


class Agent(BaseAgent):
    """A hierarchical agent class that implements a multi-level learning architecture.

    This agent class creates and manages a hierarchical structure of learning agents organized in
    2 levels (bottom, top). Each level contains one or more agents that can
    communicate and make decisions. The hierarchy allows for both independent and coordinated
    learning across different levels.

    The action from the top to the bottom level are discrete.

        env (ParallelEnv): The parallel environment that the agent will interact with.
        args (None | Args, optional): Configuration arguments for the agent. If None, default
            Args will be used. Defaults to None.

    Attributes:
        args (Args): Configuration arguments for the agent.
        device (torch.device): The device (CPU/GPU) where computations will be performed.
        env (ParallelEnv): The environment the agent interacts with.
        hierarchy (Hierarchy): The hierarchical structure containing all agent levels.
        num_env_agents (int): Number of agents in the environment.
        bottom_env_links (dict): Mapping between bottom level agents and environment agents.
        top_bottom_links (dict): Mapping between top level and bottom level agents.

    Methods:
        seed(seed): Sets random seeds for reproducibility.
        train(env, log_path, run_name): Trains the hierarchical agent.
        save_agent(save_path, name): Saves the agent's hierarchy to disk.
        load_agent(load_path, name): Loads a trained agent from disk.
        act(observation): Returns action based on given observation.

    Note:
        The hierarchy consists of three levels:
        - Bottom level: Contains individual PPO agents for each environment agent
        - Top level: Contains a single MAPPO agent that coordinates bottom level agents

    Example:
        >>> env = ParallelEnv()
        >>> args = Args()
        >>> agent = Agent(env, args)
        >>> agent.train()

        This is the structure that the agent would have for an environment with 4 agents.
        Instantiated hierarchy:
        └── top_mappo
            ├── mappo_0
            │   └── agent_0 (env)
            ├── mappo_1
            │   └── agent_1 (env)
            ├── mappo_2
            │   └── agent_2 (env)
            └── mappo_3
                └── agent_3 (env)
    """

    def __init__(self, env: ParallelEnv, args: None | Args = None) -> None:
        if args is None:
            self.args: Args = Args()
        else:
            self.args: Args = args

        if self.args.cuda >= 0:
            self.device = torch.device(
                f"cuda:{self.args.cuda}" if torch.cuda.is_available() else "cpu"
            )
        self.env = env
        self.seed(self.args.seed)

        # Create hierarchy
        self.hierarchy = Hierarchy()
        self.num_env_agents = len(env.possible_agents)

        # ---------------------------
        # Define agent names
        name_top_agent = "top_mappo"
        bottom_agent_names = [f"agent_{i}" for i in range(self.num_env_agents)]

        # Define hierarchy links
        self.bottom_env_links = {
            agent_name: [agent_name] for agent_name in bottom_agent_names
        }
        self.top_bottom_links = {name_top_agent: bottom_agent_names}
        # ---------------------------

        # Configure bottom level
        # ================================
        # Configure agents
        bottom_agents = [
            AgentConfig(
                name=name,
                communication_space=GymDict(
                    {name: Box(-np.inf, np.inf, shape=[self.args.comm_size])}  # type: ignore
                )
                if (hasattr(self.args, "learn_comm") and self.args.learn_comm)  # type: ignore
                else GymDict({name: env.observation_space(name)}),
                agent_class=PPO,
                agent_kwargs={"args": self.args.ppo_args},
                device=self.device,
            )
            for name in bottom_agent_names
        ]

        # Configure bottom level action space
        bottom_level_act_shape = 2
        bottom_level_action_space = {}
        for hl_agent, agent_names in self.top_bottom_links.items():
            # Continuous action space!
            agent_spaces = {
                name: Box(-np.inf, np.inf, shape=[bottom_level_act_shape])
                for name in agent_names
            }
            bottom_level_action_space[hl_agent] = GymDict(agent_spaces)  # type: ignore
        bottom_level_action_space = GymDict(bottom_level_action_space)

        # Configure level
        bottom_level = LevelConfig(
            name="bottom",
            agents=bottom_agents,
            uplinks=self.top_bottom_links,
            downlinks=self.bottom_env_links,
            action_frequency=self.args.ifreq_bottom,
            trace_type="full" if self.args.save_all_trace else "reward",
            concat_obs=True,
            action_space=bottom_level_action_space,
            env=env,
        )
        self.hierarchy.add_level_config(bottom_level)
        # ================================

        # Configure top level
        # ================================
        top_agent = AgentConfig(
            name=name_top_agent,
            communication_space=None,
            agent_class=MAPPO,
            agent_kwargs={"args": self.args.mappo_args},
            device=self.device,
        )

        top_level = LevelConfig(
            name="top",
            agents=[top_agent],
            uplinks=None,
            downlinks=self.top_bottom_links,
            action_frequency=self.args.ifreq_top,
            trace_type="full" if self.args.save_all_trace else "reward",
            env=self.hierarchy.levels[-1],
        )
        self.hierarchy.add_level_config(top_level)
        # ================================

        print("Instantiated hierarchy:")
        self.hierarchy.print_tree()

    def seed(self, seed):
        """
        Sets random seeds for reproducibility across different libraries.

        Args:
            seed (int): The random seed value to be used for random number generation.

        Note:
            This method sets seeds for the following libraries:
            - Python's random module
            - NumPy's random number generator
            - PyTorch's random number generator
            It also sets PyTorch's CUDNN backend to deterministic mode based on args.torch_deterministic.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def train(
        self,
        env: ParallelEnv | None = None,
        log_path: Path | str | None = None,
        run_name: str | None = None,
    ):
        """
        Train the hierarchical agent using the hierarchy.

        This method implements the training loop for the hierarchical agent. It sets up logging,
        initializes the environment, and runs the training episodes until the specified total
        timesteps are reached. Creates a directory structure for logging under log_path/run_name,
        saves hyperparameters and interface level information as JSON files, uses tensorboard
        for logging training metrics. Training continues until total_timesteps (specified in
        self.args) is reached.

        Args:
            env (ParallelEnv | None, optional): The environment to train on. If None, uses
                the environment stored in self.env passed at init. Defaults to None.
            log_path (Path | str | None, optional): Path where to save the training logs.
                If None, saves to "runs" directory. Defaults to None.
            run_name (str | None, optional): Name for the training run. If None, generates
                a name using experiment name, seed and timestamp. Defaults to None.

        Returns:
            None

        Note:
            - Creates a directory structure for logging under log_path/run_name
            - Saves hyperparameters and interface level information as JSON files
            - Uses tensorboard for logging training metrics
            - Training continues until total_timesteps (specified in self.args) is reached
        """
        if env is None:
            env = self.env
        self.hierarchy.connect(env)

        # Prepare for logging
        if run_name is None:
            run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        if log_path is None:
            log_path = Path("runs")
        else:
            log_path = Path(log_path)

        if not (log_path / run_name).exists():
            os.makedirs(log_path / run_name)

        args_dict = self.args.to_dict()  # type: ignore
        with open(log_path / run_name / "params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

        interface_level_name = {"name": self.hierarchy.interface_level.name}
        with open(log_path / run_name / "interface_level.json", "w") as f:
            json.dump(interface_level_name, f, indent=4)

        save_path = log_path / run_name / "training"

        writer = SummaryWriter(save_path / "tboard")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )

        self.hierarchy.set_logger(logger=writer, save_path=save_path)

        # Training loop
        # ---------------------------
        done = False
        episode = 0
        self.hierarchy.reset()

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            if done:
                done = False
                self.hierarchy.reset()
                episode += 1

            # Training and environment interaction is handled by the hierarchy
            _, reward, terminated, truncated, _ = self.hierarchy.step(action=None)

            if any(terminated.values()) or any(truncated.values()):
                done = True
        # ---------------------------

        # Final cleanup
        self.hierarchy.reset()
        writer.close()

    def save_agent(self, save_path: str | Path, name: None | str = None):
        """
        Save the agent's hierarchy to disk.

        This method saves the entire hierarchy structure of the agent to the specified path.

        Args:
            save_path (str | Path): The directory path where to save the agent's hierarchy.
            name (str | None, optional): The name to use for saving. If None, uses default
                naming. Defaults to None.

        Returns:
            None
        """
        self.hierarchy.save(Path(save_path))

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """Load a trained agent from disk.

        This method loads a previously trained agent's model from the specified path.

        Args:
            load_path (Path | str): The directory path where the model is saved.
            name (str, optional): Name of the model file. Defaults to "trained_model".

        Returns:
            bool: True if the agent was loaded successfully.

        Raises:
            RuntimeError: If the agents could not be loaded from the specified path.

        Example:
            >>> agent.load_agent("/path/to/model", "my_model")
            Agents loaded successfully
            True
        """
        loaded = self.hierarchy.load(Path(load_path))
        if loaded:
            print("Agents loaded successfully")
        else:
            raise RuntimeError("Could not load agents!")
        return loaded

    def act(self, observation: Dict[str, np.ndarray]) -> dict:
        """
        Takes an observation and returns the action decided by the hierarchy.

        Args:
            observation (Dict[str, np.ndarray]): A dictionary containing the current observation
                of the environment, where keys are observation names and values are numpy arrays.

        Returns:
            dict: The action to be taken, determined by the hierarchy's decision-making process.
        """
        return self.hierarchy.act(observation)
