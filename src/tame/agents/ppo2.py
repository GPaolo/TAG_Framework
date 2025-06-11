from tame.hierarchy import Hierarchy, LevelConfig, AgentConfig
from tame.agents.monolithic_ppo import Agent as PPO
from tame.hierarchy import BaseAgent
import os
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from functools import cached_property
from tame.utils.utils import filter_unexpected_fields
from pettingzoo import ParallelEnv
from gymnasium.spaces import Dict as GymDict, Box, Discrete
from typing import Dict
import numpy as np
from pathlib import Path
import json
import torch
import random
import time
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tame.utils.config import ArgsInterface


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
    save_all_trace: bool = False
    total_timesteps: int = 500000
    learning_rate: float = 0.001
    gamma: float = 0.99
    anneal_lr: bool = True
    gae_lambda: float = 0.95
    batch_size: int = 2048 * 1
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    @cached_property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class Agent(BaseAgent):
    """A hierarchical agent implementation using PPO (Proximal Policy Optimization).

    This agent creates a two-level hierarchy with multiple PPO agents:
    - A bottom level with individual PPO agents for each environment agent
    - A top level with a single PPO agent coordinating the bottom level agents

    The hierarchy allows for coordinated decision making across multiple agents while
    maintaining individual agent policies.

    Args:
        env (ParallelEnv): The parallel environment that the agent will interact with
        args (Args | None): Configuration arguments for the agent. If None, default Args() will be used

    Attributes:
        args (Args): Configuration arguments for the agent
        device (torch.device): Device to run computations on (CPU/GPU)
        hierarchy (Hierarchy): The hierarchical structure of agents
        num_env_agents (int): Number of agents in the environment
        env_obs_sizes (list): List of observation space sizes for each agent
        env_actions_ns (list): List of action space sizes for each agent
        bottom_env_links (dict): Mapping of bottom level agents to environment agents
        top_bottom_links (dict): Mapping of top level agent to bottom level agents

    Methods:
        seed(seed): Sets random seeds for reproducibility
        get_spaces(env): Extracts observation and action spaces from environment
        train(env, log_path, run_name): Trains the hierarchical agent
        save_agent(save_path, name): Saves the trained agent
        load_agent(load_path, name): Loads a trained agent
        act(observation): Returns actions based on current observation

    Example:
        This is the structure that the agent would have for an environment with 4 agents:
        Instantiated hierarchy:
        └── top_ppo
            ├── ppo_0
            │   └── agent_0 (env)
            ├── ppo_1
            │   └── agent_1 (env)
            ├── ppo_2
            │   └── agent_2 (env)
            └── ppo_3
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
        self.seed(self.args.seed)
        self.env = env

        # Create builder
        self.hierarchy = Hierarchy()
        self.num_env_agents = len(env.possible_agents)
        self.get_spaces(env)

        # Describe hierarchy links
        # ---------------------------
        bottom_agent_names = [f"agent_{i}" for i in range(self.num_env_agents)]
        top_agent_name = "top_ppo"

        # Links between bottom level and environment agents
        self.bottom_env_links = {
            agent_name: [agent_name] for agent_name in bottom_agent_names
        }

        self.top_bottom_links = {top_agent_name: bottom_agent_names}
        # ---------------------------

        # Configure bottom level
        # ================================
        # Configure bottom level agents
        # ---------------------------
        bottom_agents = [
            AgentConfig(
                name=bottom_agent_names[i],
                communication_space=GymDict(
                    {f"agent_{i}": Box(np.inf, np.inf, shape=[self.env_obs_sizes[i]])}
                ),
                agent_class=PPO,
                agent_kwargs={"args": args},
                device=self.device,
            )
            for i in range(self.num_env_agents)
        ]
        # ---------------------------

        # Config bottom level env
        # ---------------------------
        # Make custom action space for bottom level
        bottom_lev_action_space = {}
        for hl_agent, agent_names in self.top_bottom_links.items():
            agent_spaces = {}
            for agent_name in agent_names:
                agent_spaces[agent_name] = Discrete(5)
            bottom_lev_action_space[hl_agent] = GymDict(agent_spaces)
        bottom_lev_action_space = GymDict(bottom_lev_action_space)

        # Add level to hierarchy
        self.hierarchy.add_level_config(
            LevelConfig(
                name="bottom",
                agents=bottom_agents,
                uplinks=self.top_bottom_links,
                downlinks=self.bottom_env_links,
                action_frequency=1,
                trace_type="full" if self.args.save_all_trace else "reward",
                concat_obs=True,
                action_space=bottom_lev_action_space,
                env=env,
            )
        )
        # ---------------------------
        # ================================

        # Config top level
        # ================================
        # Configure top level agent
        top_agent = AgentConfig(
            name=top_agent_name,
            communication_space=None,
            agent_class=PPO,
            agent_kwargs={"args": args},
            device=self.device,
        )

        self.hierarchy.add_level_config(
            LevelConfig(
                name="top",
                agents=[top_agent],
                uplinks=None,
                downlinks=self.top_bottom_links,
                action_frequency=1,
                trace_type="full" if self.args.save_all_trace else "reward",
                env=self.hierarchy.levels[-1],
            )
        )

        print("Instantiated hierarchy:")
        self.hierarchy.print_tree()

    def seed(self, seed):
        """
        Sets random seeds for reproducibility across different random number generators.

        This method initializes random seeds for Python's random module, NumPy, and PyTorch
        to ensure reproducible results across multiple runs.

        Args:
            seed (int): The random seed value to use for initialization.

        Note:
            Also sets PyTorch's CUDNN backend to deterministic mode based on args setting.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def get_spaces(self, env):
        """Get observation and action spaces from environment.

        This method extracts the observation and action space dimensions for each agent
        from the environment and stores them in class variables.

        Args:
            env: The environment object containing observation_spaces and action_spaces dictionaries.
                Each space corresponds to an agent.

        Attributes set:
            env_obs_sizes (list): List of observation space dimensions for each agent.
            env_actions_ns (list): List of discrete action space sizes for each agent.
        """
        self.env_obs_sizes = []
        for agent in env.observation_spaces:
            obs_space = env.observation_spaces[agent]
            self.env_obs_sizes.append(obs_space.shape[0])

        self.env_actions_ns = []
        for agent in env.action_spaces:
            act_space = env.action_spaces[agent]
            self.env_actions_ns.append(act_space.n)

    def train(
        self,
        env: ParallelEnv | None = None,
        log_path: Path | str | None = None,
        run_name: str | None = None,
    ):
        """Train the hierarchical agent using PPO.

        This method trains the hierarchical agent using the PPO algorithm. It handles environment setup,
        logging configuration, and executes the main training loop.

        Args:
            env (ParallelEnv | None, optional): The environment to train in. If None, uses the agent's default environment.
                Defaults to None.
            log_path (Path | str | None, optional): Directory path for saving logs. If None, saves to "runs" directory.
                Defaults to None.
            run_name (str | None, optional): Name for the training run. If None, generates name from experiment parameters.
                Defaults to None.

        Returns:
            None

        Notes:
            - Creates directories for logging training metrics and parameters
            - Sets up TensorBoard logging
            - Executes training loop for specified number of timesteps
            - Handles episode resets and termination conditions
            - Saves hyperparameters and interface level configuration
        """
        if env is None:
            env = self.env

        # Connect hierarchy to the environment
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

            _, reward, terminated, truncated, _ = self.hierarchy.step(action=None)

            if any(terminated.values()) or any(truncated.values()):
                done = True

        # Final cleanup
        self.hierarchy.reset()

    def save_agent(self, save_path: str | Path, name: None | str = None):
        """
        Saves the agent's hierarchy to a specified path.

        Args:
            save_path (str | Path): The directory path where the agent's hierarchy will be saved.
            name (str | None, optional): Optional name identifier for the saved agent. Defaults to None.

        Note:
            The function saves the complete hierarchical structure of the agent to the specified path.
        """
        self.hierarchy.save(Path(save_path))

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """Load agent's parameters from a file.

        This method loads the hierarchical agent's parameters from a specified path.

        Args:
            load_path (Union[Path, str]): Path to the directory containing the saved model.
            name (str, optional): Name of the saved model. Defaults to "trained_model".

        Returns:
            bool: True if loading was successful.

        Raises:
            RuntimeError: If the agents could not be loaded.

        Example:
            >>> agent.load_agent("path/to/saved/model")
        """
        loaded = self.hierarchy.load(Path(load_path))
        if loaded:
            print("Agents loaded successfully")
        else:
            raise RuntimeError("Could not load agents!")
        return loaded

    def act(self, observation: Dict[str, np.ndarray]) -> dict:
        """
        Performs an action based on the given observation through the hierarchy.

        Args:
            observation (Dict[str, np.ndarray]): A dictionary containing the observations of the environment,
                where keys are observation names and values are numpy arrays containing the observation data.

        Returns:
            dict: A dictionary containing the action to be taken by the agent.
        """
        return self.hierarchy.act(observation)
