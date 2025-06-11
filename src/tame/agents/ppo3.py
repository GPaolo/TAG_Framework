from tame.hierarchy import Hierarchy, LevelConfig, AgentConfig
from tame.agents.monolithic_ppo import Agent as PPO
from tame.hierarchy import BaseAgent
import os
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from functools import cached_property
from tame.utils.utils import filter_unexpected_fields
from tame.utils.space_utils import merge_spaces
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
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.015
    ifreq_bottom: int = 1
    ifreq_mid: int = 1
    ifreq_top: int = 1
    learn_comm: bool = False
    ae_epochs: int = 100
    comm_size: int = 16

    @cached_property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class Agent(BaseAgent):
    """A hierarchical agent implementation using PPO.

    This class implements a three-level hierarchical agent structure where multiple bottom-level
    agents are controlled by middle-level agents, which in turn are controlled by a top-level agent.
    The hierarchy is configured to handle parallel environments and supports optional communication
    between agents.

    Args:
        env (ParallelEnv): The parallel environment the agent will interact with.
        args (None | Args, optional): Configuration arguments for the agent. If None, default Args will be used.
            Defaults to None.

    Attributes:
        args (Args): Configuration arguments for the agent.
        device (torch.device): The device (CPU/GPU) where the computations will be performed.
        hierarchy (Hierarchy): The hierarchical structure of the agent.
        num_env_agents (int): Number of agents in the environment.
        bottom_env_links (dict): Mapping of bottom-level agents to environment agents.
        mid_bottom_links (dict): Mapping of middle-level agents to bottom-level agents.
        top_mid_links (dict): Mapping of top-level agent to middle-level agents.

    Examples:
        >>> env = ParallelEnv()
        >>> agent = Agent(env)
        >>> agent.train(env)
        >>> action = agent.act(observation)

    Note:
        - The hierarchy consists of three levels: bottom, middle, and top.
        - Each level can be configured with different action frequencies.
        - Supports optional learned communication between agents.
        - Uses PPO based agents as sub-agents in each level.

    Example:
        This is the structure that the agent would have for an environment with 4 agents.
        Instantiated hierarchy:
        └── top_ppo
            ├── middle_ppo_1
            │   ├── agent_0
            │   │   └── agent_0 (env)
            │   └── agent_1
            │       └── agent_1 (env)
            └── middle_ppo_2
                ├── agent_2
                │   └── agent_2 (env)
                └── agent_3
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

        # Describe hierarchy links
        # ---------------------------
        bottom_agent_names = [f"agent_{i}" for i in range(self.num_env_agents)]
        middle_names = ["middle_ppo_1", "middle_ppo_2"]
        top_agent_name = "top_ppo"

        # Links between bottom level and environment agents
        self.bottom_env_links = {
            agent_name: [agent_name] for agent_name in bottom_agent_names
        }

        self.mid_bottom_links = {
            middle_names[0]: bottom_agent_names[: int(self.num_env_agents / 2)],
            middle_names[1]: bottom_agent_names[int(self.num_env_agents / 2) :],
        }

        self.top_mid_links = {top_agent_name: middle_names}
        # ---------------------------

        # Configure bottom level
        # ================================
        # Config level agents and their communication space
        bottom_agents = [
            AgentConfig(
                name=name,
                communication_space=GymDict(
                    {name: Box(-np.inf, np.inf, shape=[self.args.comm_size])}
                )
                if (hasattr(self.args, "learn_comm") and self.args.learn_comm)
                else GymDict({name: env.observation_space(name)}),
                agent_class=PPO,
                agent_kwargs={"args": args},
                device=self.device,
            )
            for name in bottom_agent_names
        ]

        # Config bottom level action space
        bottom_lev_action_space = {}
        for hl_agent, agent_names in self.mid_bottom_links.items():
            agent_spaces = {agent_name: Discrete(5) for agent_name in agent_names}
            bottom_lev_action_space[hl_agent] = GymDict(agent_spaces)  # type: ignore
        bottom_lev_action_space = GymDict(bottom_lev_action_space)

        self.hierarchy.add_level_config(
            LevelConfig(
                name="bottom",
                agents=bottom_agents,
                uplinks=self.mid_bottom_links,
                downlinks=self.bottom_env_links,
                action_frequency=self.args.ifreq_bottom,
                trace_type="full" if self.args.save_all_trace else "reward",
                concat_obs=True,
                action_space=bottom_lev_action_space,
                env=env,
            )
        )
        # ================================

        # Configure middle level
        # ================================
        middle_agents = []
        prev_level = self.hierarchy.levels[-1]
        for agent_name in middle_names:
            # Make comm space
            if hasattr(self.args, "learn_comm") and self.args.learn_comm:
                agent_comm_space = GymDict(
                    {agent_name: Box(-np.inf, np.inf, shape=[self.args.comm_size])}
                )
            else:
                # In this case, we just concatenate the observations from the level l-1
                prev_lev_obs_spaces = list(
                    prev_level.observation_space(agent_name).values()  # type: ignore
                )
                agent_comm_space = GymDict(
                    {agent_name: merge_spaces(prev_lev_obs_spaces)}
                )

            # Agent config
            middle_agents.append(
                AgentConfig(
                    name=agent_name,
                    communication_space=agent_comm_space,
                    agent_class=PPO,
                    agent_kwargs={"args": args},
                    device=self.device,
                )
            )

        # Config middle level action space
        mid_lev_action_space = {}
        for hl_agent, agent_names in self.top_mid_links.items():
            agent_spaces = {}
            for agent_name in agent_names:
                agent_spaces[agent_name] = Discrete(5)
            mid_lev_action_space[hl_agent] = GymDict(agent_spaces)
        mid_lev_action_space = GymDict(mid_lev_action_space)

        self.hierarchy.add_level_config(
            LevelConfig(
                name="middle",
                agents=middle_agents,
                uplinks=self.top_mid_links,
                downlinks=self.mid_bottom_links,
                action_frequency=self.args.ifreq_mid,
                trace_type="full" if self.args.save_all_trace else "reward",
                concat_obs=True,
                action_space=mid_lev_action_space,
                env=prev_level,
            )
        )
        # ================================

        # Configure top level
        # ================================
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
                downlinks=self.top_mid_links,
                action_frequency=self.args.ifreq_top,
                trace_type="full" if self.args.save_all_trace else "reward",
                env=self.hierarchy.levels[-1],
            )
        )
        # ================================

        print("Instantiated hierarchy:")
        self.hierarchy.print_tree()

        print("Detailed hierarchy:")
        self.hierarchy.print_hierarchy_details()

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
