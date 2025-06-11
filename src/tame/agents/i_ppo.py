from tame.hierarchy import Hierarchy, LevelConfig, AgentConfig
from tame.agents.monolithic_ppo import Agent as PPO
from tame.hierarchy import BaseAgent
import os
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from functools import cached_property
from tame.utils.utils import filter_unexpected_fields
from pettingzoo import ParallelEnv
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
    cuda: int = 0
    save_model: bool = True
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    anneal_lr: bool = True
    gae_lambda: float = 0.95
    batch_size: int = 2048 * 1  # It's num_steps * num_envs (I don't make parallel envs)
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    save_all_trace: bool = False
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    verbose: bool = True

    @cached_property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class Agent(BaseAgent):
    """Implementation of Independet PPO (I-PPO) agent.

    This agent class implements a 1 level hierarchy (with a single bottom level) containing
    independent PPO agents.

    Args:
        env (ParallelEnv): The parallel environment the agent will interact with
        args (Args | None, optional): Configuration arguments for the agent. Defaults to None.

    Attributes:
        args (Args): Configuration arguments for the agent
        device (torch.device): Device to run computations on (CPU/GPU)
        hierarchy (Hierarchy): The hierarchical structure containing the agents
        num_env_agents (int): Number of agents in the environment
        bottom_env_links (dict): Mapping of bottom level agent names to their environment links

    Methods:
        seed(seed): Sets random seeds for reproducibility
        train(env, log_path, run_name): Trains the hierarchical agent
        save_agent(save_path, name): Saves the agent's state
        load_agent(load_path, name): Loads the agent's state
        act(observation): Returns actions based on given observations

    Example:
        >>> env = ParallelEnv()
        >>> agent = Agent(env)
        >>> agent.train(env)
    """

    def __init__(self, env: ParallelEnv, args: None | Args = None) -> None:
        if args is None:
            self.args: Args = Args()
        else:
            self.args: Args = args

        self.device = torch.device("cpu")
        if self.args.cuda >= 0:
            self.device = torch.device(
                f"cuda:{self.args.cuda}" if torch.cuda.is_available() else "cpu"
            )
        self.seed(self.args.seed)

        # Create hierarchy structure
        self.hierarchy = Hierarchy()
        self.num_env_agents = len(env.possible_agents)
        self.env = env

        # As this is I-PPO, it only consists of one level directly interfaced with the environemnt
        bottom_agents = [
            AgentConfig(
                name=f"agent_{i}",
                communication_space=None,
                agent_class=PPO,
                agent_kwargs={"args": args},
                device=self.device,
            )
            for i in range(len(env.possible_agents))
        ]

        bottom_agent_names = [
            f"agent_{agent_idx}" for agent_idx in range(self.num_env_agents)
        ]

        # Connections between the PPO agents and the agents in the environment
        self.bottom_env_links = {
            agent_name: [agent_name] for agent_name in bottom_agent_names
        }

        # Add configuration of the level to the hierarchy
        self.hierarchy.add_level_config(
            LevelConfig(
                name="bottom",
                agents=bottom_agents,
                uplinks=None,
                downlinks=self.bottom_env_links,
                action_frequency=1,
                trace_type="full" if self.args.save_all_trace else "reward",
                concat_obs=False,
                env=env,
            )
        )

        # Build hierarchy
        print("Instantiated hierarchy:")
        self.hierarchy.print_tree()

    def seed(self, seed):
        """
        Sets random seeds for reproducibility across different random number generators.

        Args:
            seed (int): The seed value to use for random number generation.

        Effects:
            - Sets seed for Python's random module
            - Sets seed for NumPy's random number generator
            - Sets seed for PyTorch's random number generator
            - Configures PyTorch's CUDNN backend determinism based on args.torch_deterministic

        Note:
            This ensures reproducible behavior across different runs when using random operations
            in Python, NumPy and PyTorch.
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
        """Train the I-PPO agent in the given environment.

        This method trains the agent. It handles logging setup,
        training loop execution, and episode management.

        Args:
            env (ParallelEnv | None): The environment to train in. Must be compatible with the agent's structure.
                If None, defaults to using the env given at agent init. Defaults to None.
            log_path (Path | str | None, optional): Path where to save logs and checkpoints.
                If None, defaults to "runs". Defaults to None.
            run_name (str | None, optional): Name for the training run. If None, generates
                a name using experiment name, seed and timestamp. Defaults to None.

        Returns:
            None

        Notes:
            - Sets up logging directories and tensorboard writer
            - Saves hyperparameters and interface level configuration
            - Executes training loop for specified number of timesteps
            - Handles episode resets and termination conditions
            - Logs training progress via tqdm
        """
        if env is None:
            env = self.env

        # Connect the environment to the hierarchy
        self.hierarchy.connect(env)

        # Prepare for logging
        # ---------------------------
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

        # Needed so the plotting can load the right traces
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
        # ---------------------------

        # Start.
        self.hierarchy.reset()
        done = False
        episode = 0

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if done:
                # We don't have to reset the env, just the topmost level
                done = False
                self.hierarchy.reset()
                episode += 1
            # ---------------------------

            _, reward, terminated, truncated, _ = self.hierarchy.step(action=None)

            if any(terminated.values()) or any(truncated.values()):
                done = True

        # To save the last trace
        self.hierarchy.reset()

    def save_agent(self, save_path: str | Path, name: None | str = None):
        """Save the agent and hierarchy to a file.

        This method saves the complete agent including the learned hierarchy to the specified path.

        Args:
            save_path (str | Path): Path where to save the agent.
            name (str | None, optional): Name for the saved file. Defaults to None.

        Returns:
            None

        Example:
            >>> agent.save_agent("/path/to/save")
            >>> agent.save_agent(Path("/path/to/save"), name="my_agent")
        """
        self.hierarchy.save(Path(save_path))

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """
        Load a trained agent from a file.

        This method loads a trained hierarchical agent from the specified path.

        Args:
            load_path (Union[Path, str]): Path to the directory containing the saved model.
            name (str, optional): Name of the model file. Defaults to "trained_model".

        Returns:
            bool: True if loading was successful.

        Raises:
            RuntimeError: If the bottom level of the hierarchy could not be loaded.
        """
        loaded = self.hierarchy.load(Path(load_path))
        if loaded:
            print("Bottom level loaded")
        else:
            raise RuntimeError("Could not load bottom level!")
        return loaded

    def act(self, observation: Dict[str, np.ndarray]) -> dict:
        """
        Apply the agent's policy to select an action based on the current observation.

        Args:
            observation (Dict[str, np.ndarray]): A dictionary containing the current observation of the environment,
                with string keys and numpy array values.

        Returns:
            dict: A dictionary containing the selected action and any additional information
                from the hierarchical policy.
        """
        return self.hierarchy.act(observation)
