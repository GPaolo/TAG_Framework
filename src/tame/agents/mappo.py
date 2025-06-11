import os
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

# from tqdm import tqdm
from pathlib import Path
from dataclasses_json import dataclass_json
from tame.hierarchy.base_agent import LevelAgent
from gymnasium.spaces import Box, Discrete
from gymnasium.spaces import Dict as GymDict

# from tame.data_handling.trace import Trace, RewardTrace
from tame.external_algo.mappo import MAPPO_MPE
import json
from dataclasses import dataclass
from typing import List, Dict
from pettingzoo import ParallelEnv

from tame.external_algo.mappo_trainer import ReplayBuffer
from types import SimpleNamespace
from tqdm import tqdm
from tame.data_handling.trace import Trace, RewardTrace
from tame.utils.config import ArgsInterface


@dataclass_json
@dataclass
class Args(ArgsInterface):
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int | None = 1
    torch_deterministic: bool = True
    cuda: int = 0
    save_model: bool = True
    verbose: bool = False
    save_all_trace: bool = False
    total_timesteps: int = 500000
    lr: float = 0.001  # Changed from learning_rate
    episode_length: int = 25
    gamma: float = 0.99
    lamda: float = 0.95
    epsilon: float = 0.2
    K_epochs: int = 10
    mini_batch_size: int = 64
    batch_size: int = 400
    entropy_coef: float = 0.01
    mlp_hidden_dim: int = 64
    use_rnn: bool = False
    rnn_hidden_dim: int = 64
    use_relu: bool = True
    use_orthogonal_init: bool = True
    set_adam_eps: bool = True
    use_grad_clip: bool = True
    use_lr_decay: bool = True
    use_adv_norm: bool = True
    use_value_clip: bool = True
    add_agent_id: bool = True
    save_all_trace: bool = False


class Agent(LevelAgent):
    """Multi-Agent Proximal Policy Optimization (MAPPO) agent implementation.

    This class implements a MAPPO agent that can train and act in multi-agent environments.
    It inherits from LevelAgent (so it can be used in a hierarchical agent, but can also be used as a standalone algorithm)
    and implements policy optimization using the MAPPO algorithm.
    It is mainly a wrapper around the MAPPO implementation in `external_algo/mappo.py`.

    Args:
        observation_space (GymDict): The observation space for each agent in the environment
        action_space (GymDict): The action space for each agent in the environment
        device: The device (CPU/GPU) to run computations on
        communication_space (GymDict, optional): Communication space if agents can communicate. Defaults to None.
        name (str, optional): Name identifier for the agent. Defaults to "mappo".
        args (Args, optional): Configuration arguments for the agent. Defaults to None.
        torch_compile (bool, optional): Whether to compile model with torch.compile(). Defaults to False.

    Attributes:
        observation_space (GymDict): Stored observation space
        action_space (GymDict): Stored action space
        communication_space (GymDict): Stored communication space
        args (Args): Configuration arguments
        device: Device for computations
        name (str): Agent identifier
        mappo_args (SimpleNamespace): Arguments specific to MAPPO algorithm
        mappo (MAPPO_MPE): MAPPO policy network
        replay_buffer (ReplayBuffer): Buffer for storing experience
        trained (bool): Whether agent has been trained

    Methods:
        get_spaces(): Processes observation and action spaces
        seed(seed): Sets random seeds
        act(observation): Selects actions in evaluation mode
        act_train(observation, global_step): Selects actions in training mode
        store(state, action, reward, done): Stores transitions in replay buffer
        update_step(global_step, writer): Performs training updates
        train(env, log_path, run_name): Trains the agent
        save_agent(save_path, name): Saves trained model
        load_agent(load_path, name): Loads trained model

    Notes:
        As it respects the LevelAgent interface, it can be used in hierarchical learning setups.
        It can also be used as a standalone agent.
    """

    def __init__(
        self,
        observation_space: GymDict,
        action_space: GymDict,
        device,
        communication_space: GymDict | None = None,
        name: str = "mappo",
        args: Args | None = None,
        torch_compile: bool = False,
    ) -> None:
        # Store spaces as required by LevelAgent
        self.observation_space = observation_space
        self.action_space = action_space
        self.communication_space = communication_space

        if args is not None:
            self.args = args
        else:
            self.args = Args()

        self.device = device
        self.name = name

        # env setup
        self.get_spaces()

        # Setup MAPPO args
        self.mappo_args = SimpleNamespace()
        self.mappo_args.N = self.num_env_agents
        self.mappo_args.discrete_actions = not self.continuous_actions
        self.mappo_args.action_dim = self.actions_dim
        self.mappo_args.obs_dim = self.obs_size_single
        self.mappo_args.state_dim = self.obs_size
        self.mappo_args.episode_limit = self.args.episode_length
        self.mappo_args.torch_compile = torch_compile

        self.name = name
        self.trained = False

        # Copy MAPPO-specific args
        for attr in [
            "rnn_hidden_dim",
            "mlp_hidden_dim",
            "batch_size",
            "mini_batch_size",
            "lr",
            "gamma",
            "lamda",
            "epsilon",
            "K_epochs",
            "entropy_coef",
            "set_adam_eps",
            "use_grad_clip",
            "use_lr_decay",
            "use_adv_norm",
            "use_rnn",
            "add_agent_id",
            "use_value_clip",
            "use_relu",
            "use_orthogonal_init",
        ]:
            setattr(self.mappo_args, attr, getattr(self.args, attr))

        self.mappo_args.max_train_steps = self.args.total_timesteps
        self.mappo_args.device = self.device
        self.mappo_args.name = self.name

        self.mappo = MAPPO_MPE(self.mappo_args)
        self.replay_buffer = ReplayBuffer(self.mappo_args)
        self.current_episode = []
        self.episodes = []

    def get_spaces(self):
        """
        Computes and sets observation and action space dimensions and types from environment spaces.

        This method processes the observation and action spaces for all agents to determine:
        - Total observation size across all agents
        - Individual agent observation size
        - Number of agents in environment
        - Action space dimensionality
        - Whether actions are continuous or discrete

        Raises:
            ValueError: If action space is neither Discrete nor Box type from gymnasium

        Note:
            Expects self.observation_space and self.action_space to be set before calling.
            Sets the following instance attributes:
                - self.obs_size: Combined size of observations across all agents
                - self.obs_size_single: Size of single agent observation
                - self.num_env_agents: Total number of agents
                - self.actions_dim: Dimension of action space
                - self.continuous_actions: Boolean indicating if actions are continuous
        """
        self.obs_size = 0
        self.num_env_agents = 0
        for agent in self.observation_space:
            obs_space = self.observation_space[agent]
            self.obs_size += obs_space.shape[0]  # type: ignore
            self.obs_size_single = obs_space.shape[0]  # type: ignore
            self.num_env_agents += 1

        self.actions_dim = 0
        self.continuous_actions = False
        for agent in self.action_space:
            act_space = self.action_space[agent]
            if isinstance(act_space, Box):
                self.continuous_actions = True
                self.actions_dim = max(self.actions_dim, act_space.shape[0])
            elif isinstance(act_space, Discrete):
                self.actions_dim = max(self.actions_dim, act_space.n)  # type: ignore
            else:
                raise ValueError("Only Discrete or Box gymnasium spaces are supported")

    def seed(self, seed) -> None:
        """
        Sets random seeds for reproducibility across different random number generators.

        Args:
            seed (int): The seed value to use for random number generation.

        Note:
            This method sets seeds for Python's random module, NumPy, PyTorch CPU, and
            optionally PyTorch CUDA if available. When CUDA is available, it also sets
            the cudnn backend to deterministic mode based on args.torch_deterministic.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.args.torch_deterministic  # type: ignore

    def act(self, observation: dict) -> dict:
        """
        Takes observations from agents and returns actions using MAPPO policy.

        Args:
            observation (dict): Dictionary mapping agent IDs to their observations.

        Returns:
            dict: Dictionary mapping agent IDs to their selected actions.

        Example:
            >>> agent = MAPPOAgent()
            >>> obs = {'agent1': [1,2,3], 'agent2': [4,5,6]}
            >>> actions = agent.act(obs)
            >>> print(actions)
            {'agent1': 0, 'agent2': 1}
        """
        obs_n = np.array(list(observation.values()))
        actions, _ = self.mappo.choose_action(obs_n, evaluate=True)
        return {agent: action for agent, action in zip(observation.keys(), actions)}

    def act_train(self, observation: Dict[str, np.ndarray], global_step: int) -> dict:
        """
        Train-time action selection for MAPPO agents.

        Takes observations from all agents and returns actions using the MAPPO policy.
        Stores current log probabilities for later use in training.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary mapping agent IDs to their observations
            global_step (int): Current training step count

        Returns:
            dict: Dictionary mapping agent IDs to their selected actions

        Note:
            This method runs in training mode (evaluate=False) which enables exploration
        """
        obs_n = np.array(list(observation.values()))
        actions, logprobs = self.mappo.choose_action(obs_n, evaluate=False)
        self.current_logprobs = logprobs
        return {agent: action for agent, action in zip(observation.keys(), actions)}

    def store(
        self,
        state: dict,
        action: dict | None = None,
        reward: float | None = None,
        done: bool | None = None,
    ):
        """Store a transition in the replay buffer.

        This method stores transitions or final values in a replay buffer for training.
        It handles both regular transitions during episodes
        and final values at episode termination.

        Args:
            state (dict): Current state/observation for each agent
            action (dict, optional): Actions taken by each agent. None if storing final values
            reward (float, optional): Reward received. None if storing final values
            done (bool, optional): Whether episode terminated. None if storing final values

        Notes:
            - If action is None, only stores the final value for the current state
            - For regular transitions, stores the full transition tuple including:
                - observations
                - flattened state
                - values
                - actions
                - action log probabilities
                - rewards
                - done flags
            - Maintains an internal step counter for episode tracking
        """
        # Store transition in replay buffer
        if not hasattr(self, "step_count"):
            self.step_count = 0

        obs_n = np.array(list(state.values()))
        s = obs_n.flatten()
        if action is None:
            v_n = self.mappo.get_value(s)
            self.replay_buffer.store_last_value(episode_step=self.step_count, v_n=v_n)  # type: ignore
            self.step_count = 0
        else:
            a_n = np.array(list(action.values()))
            r_n = np.array([reward for _ in state.keys()])
            done_n = np.array([done for _ in state.keys()])
            v_n = self.mappo.get_value(s)

            self.replay_buffer.store_transition(
                episode_step=self.step_count,
                obs_n=obs_n,
                s=s,
                v_n=v_n,
                a_n=a_n,
                a_logprob_n=self.current_logprobs,
                r_n=r_n,
                done_n=done_n,
            )
            self.step_count += 1

    def update_step(self, global_step: int, writer: SummaryWriter | None):
        """Update step for training the MAPPO agent.

        This method is called during the environment step when training is enabled. If the replay buffer
        contains enough episodes (>= batch_size), it triggers the MAPPO training process and resets the buffer.

        Args:
            global_step (int): The current global step count in training
            writer (SummaryWriter | None): TensorBoard SummaryWriter instance for logging. Can be None.

        Returns:
            None

        Note:
            - Only triggers training when replay buffer has collected enough episodes
            - Resets replay buffer and step count after training
        """
        """Called during level_env.step when training=True"""
        if self.replay_buffer.episode_num >= self.args.batch_size:  # type: ignore
            self.mappo.train(self.replay_buffer, global_step, writer=writer)
            self.replay_buffer.reset_buffer()
            self.step_count = 0

    def train(
        self,
        env: ParallelEnv,
        log_path: Path | str | None = None,
        run_name: str | None = None,
    ):
        """Train the MAPPO agent on a parallel environment.

        This method implements the training loop for Multi-Agent Proximal Policy Optimization (MAPPO).
        It handles environment interactions, data collection, and agent updates.

        Args:
            env (ParallelEnv): A parallel environment instance following the PettingZoo parallel API
            log_path (Path | str | None, optional): Path where to save training logs and checkpoints.
                Defaults to None, which uses "runs" directory.
            run_name (str | None, optional): Name for the training run.
                Defaults to None, which generates a name using experiment name, seed and timestamp.

        The method:
        - Sets up logging directories and tensorboard writer
        - Creates a replay buffer for storing transitions
        - Runs episodes collecting experience until total_timesteps is reached
        - Periodically trains the MAPPO agent using collected experience
        - Logs training metrics and saves episode traces

        Training continues until the total number of environment steps reaches args.total_timesteps.
        Episode traces and tensorboard logs are saved in the specified log directory.

        Returns:
            None: The method updates the agent in-place and sets self.trained to True upon completion.
        """
        if run_name is None:
            run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"  # type: ignore

        if log_path is None:
            log_path = Path("runs")
        else:
            log_path = Path(log_path)

        if not (log_path / run_name).exists():
            os.makedirs(log_path / run_name)

        args_dict = self.args.to_dict()  # type: ignore
        with open(log_path / run_name / "params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

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
        total_steps = 0

        # Create replay buffer
        replay_buffer = ReplayBuffer(self.mappo_args)
        self.agent_names = []

        if self.args.save_all_trace:  # type: ignore
            trace = Trace()
        else:
            trace = RewardTrace()

        pbar = tqdm(
            total=self.args.total_timesteps,  # type: ignore
            desc="Training step:",
        )
        episode_counter = -1
        while total_steps < self.args.total_timesteps:  # type: ignore
            # Run episode
            observation, _ = env.reset()
            if len(self.agent_names) == 0:
                self.agent_names = list(observation.keys())
            episode_reward = 0

            obs_n = np.array(list(observation.values()))
            episode_counter += 1
            for episode_step in range(self.args.episode_length):  # type: ignore
                # Get actions and values
                a_n, a_logprob_n = self.mappo.choose_action(obs_n, evaluate=False)
                s = obs_n.flatten()
                v_n = self.mappo.get_value(s)

                # Convert actions to dict
                actions = {agent: a_n[i] for i, agent in enumerate(self.agent_names)}

                # Environment step
                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                # Process rewards and dones
                r_n = np.array([rewards[agent] for agent in self.agent_names])
                done_n = np.array(
                    [
                        terminated[agent] or truncated[agent]
                        for agent in self.agent_names
                    ]
                )

                # Store transition
                replay_buffer.store_transition(
                    episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
                )

                trace.add(
                    actions=actions,
                    observations=observation,
                    rewards=rewards,
                    terminations=terminated,
                    truncations=truncated,
                    infos=infos,
                    episode=episode_counter,
                )

                observation = next_obs
                obs_n = np.array(list(observation.values()))
                episode_reward += np.mean(r_n)
                total_steps += 1
                pbar.update(1)

                if all(done_n):
                    break

            # Store last value
            s = obs_n.flatten()
            v_n = self.mappo.get_value(s)
            replay_buffer.store_last_value(episode_step + 1, v_n)  # type: ignore

            if hasattr(trace, "add_final_obs"):
                trace.add_final_obs(next_obs)  # type: ignore
            trace.save_trace(save_path=save_path, episode=str(episode_counter))
            trace.empty()

            # Train when buffer is ready
            if replay_buffer.episode_num == self.args.batch_size:  # type: ignore
                self.mappo.train(replay_buffer, total_steps)
                replay_buffer.reset_buffer()

            writer.add_scalar("charts/episodic_reward", episode_reward, total_steps)

        writer.close()
        self.trained = True

    def _prepare_batch(self, episodes: List[List[Dict]]) -> dict:
        """Prepares a batch of experiences for training from collected episodes.

        This method processes a list of episodes into a batch dictionary containing observations,
        states, actions, rewards, done flags and value estimates for all agents. The final batch
        is converted to PyTorch tensors.

        Args:
            episodes (List[List[Dict]]): List of episodes, where each episode is a list of
                timesteps containing dictionaries with experience data.

        Returns:
            dict: A dictionary containing the processed batch data with the following keys:
                - obs_n: Observations for all agents
                - s: Global states
                - a_n: Actions taken by all agents
                - r_n: Rewards received by all agents
                - done_n: Done flags for all agents (all zeros since per-agent dones aren't tracked)
                - v_n: Value estimates for each state including a zero terminal value

        Note:
            The returned tensors have shape (num_episodes, episode_length, ...) where the
            remaining dimensions depend on the specific key.
        """
        batch = {"obs_n": [], "s": [], "a_n": [], "r_n": [], "done_n": [], "v_n": []}

        for episode in episodes:
            episode_obs_n = []
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_values = []

            for step in episode:
                episode_obs_n.append(step["obs_n"])
                episode_states.append(step["state"])
                episode_actions.append(step["actions"])
                episode_rewards.append(step["rewards"])
                episode_dones.append(
                    np.zeros(self.num_env_agents)
                )  # We don't track per-agent dones
                episode_values.append(self.mappo.get_value(step["state"]))

            # Add final value estimate
            episode_values.append(
                np.zeros_like(episode_values[0])
            )  # Zero for terminal state

            batch["obs_n"].append(episode_obs_n)
            batch["s"].append(episode_states)
            batch["a_n"].append(episode_actions)
            batch["r_n"].append(episode_rewards)
            batch["done_n"].append(episode_dones)
            batch["v_n"].append(episode_values)

        # Convert to torch tensors
        for key in batch:
            batch[key] = torch.FloatTensor(np.array(batch[key]))  # type: ignore

        return batch

    def save_agent(self, save_path: Path | str, name: str | None = "trained_model"):
        """
        Saves the trained MAPPO agent model to disk.

        This method creates a models subdirectory in the specified save path if it
        doesn't exist, then saves the model with the given name.

        Args:
            save_path (Path | str): Directory path where the model should be saved.
                A 'models' subdirectory will be created here.
            name (str | None, optional): Name to use for the saved model file.
                Defaults to "trained_model".

        Returns:
            None
        """
        save_path = Path(save_path) / "models"
        if not save_path.exists():
            os.makedirs(save_path)
        self.mappo.save_model(save_path=str(save_path), save_name=name)

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """
        Loads a previously saved MAPPO agent model from disk.

        This method attempts to load a model from the 'models' subdirectory in the specified
        load path. The model name extension is stripped if present.

        Args:
            load_path (Path | str): Directory path where the model should be loaded from.
                A 'models' subdirectory will be checked here.
            name (str, optional): Name of the model file to load.
                Defaults to "trained_model".

        Returns:
            bool: True if model loading was successful, False otherwise.
        """
        name = name.split(".")[0]
        load_path = Path(load_path) / "models"
        try:
            self.mappo.load_model(load_path=str(load_path), save_name=name)
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False
