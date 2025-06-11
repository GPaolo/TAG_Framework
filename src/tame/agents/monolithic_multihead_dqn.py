import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tame.data_handling.replay_buffer import ReplayBuffer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from pettingzoo import ParallelEnv
from dataclasses_json import dataclass_json
from tame.data_handling.trace import Trace, RewardTrace
from tame.hierarchy.base_agent import BaseAgent
from dataclasses import dataclass
import json
from typing import Dict
from tame.utils.config import ArgsInterface


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
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


class MultiHeadQNetwork(nn.Module):
    """Multi-headed Q-Network implementation for multi-agent deep Q-learning.

    This network architecture consists of a shared backbone network followed by
    multiple output heads, one for each agent. The shared network processes the
    observation space while each head produces Q-values for the action space of
    its corresponding agent.

    Args:
        obs_size (int): Size of the observation/state space.
        actions_n (list): List of integers where each element represents the number
            of possible actions for each agent.
        device (torch.device): Device to run the network on (CPU/GPU).
        torch_compile (bool, optional): Whether to compile the network using
            torch.compile for potential speed improvements. Defaults to True.

    Returns:
        torch.Tensor: Q-values for all agents with shape [batch_size, n_agents, n_actions].
            Where n_actions may differ for each agent according to actions_n.

    Note:
        The network architecture consists of:
        - Shared layers: Two fully connected layers (120, 84 units) with ReLU activations
        - Individual output heads: One linear layer per agent
    """

    def __init__(
        self,
        obs_size: int,
        actions_n: list,
        device: torch.device,
        torch_compile: bool = True,
    ):
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        ).to(self.device)
        self.out_layers = [
            nn.Linear(84, action).to(self.device) for action in actions_n
        ]
        if torch_compile:
            self.network = torch.compile(self.network)
            self.out_layers = nn.ModuleList(
                [torch.compile(layer) for layer in self.out_layers]  # type: ignore
            )
        else:
            self.out_layers = nn.ModuleList(self.out_layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Takes a batch of observations and processes them through the shared network and multiple output layers.

        Args:
            x (torch.Tensor): Input tensor containing observations. Shape: [batch_size, observation_dim]

        Returns:
            torch.Tensor: Q-values for each agent-action pair. Shape: [batch_size, num_agents, num_actions]

        Notes:
            - The input is first processed through a shared network
            - Then each agent's output layer processes the shared features
            - Finally dimensions are swapped to get batch first ordering
        """
        x = self.network(x)
        outs = torch.stack([layer(x) for layer in self.out_layers])
        return outs.swapaxes(0, 1)  # Swap batch and agents [batch, agents, actions]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Calculate epsilon value based on linear annealing schedule.

    This function implements a linear decay from start_e to end_e over duration steps.
    The returned epsilon is clamped to not go below end_e.

    Args:
        start_e (float): Starting epsilon value
        end_e (float): Final epsilon value
        duration (int): Number of steps over which to decay epsilon
        t (int): Current timestep

    Returns:
        float: Epsilon value at timestep t following linear schedule
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class Agent(BaseAgent):
    """A Deep Q-Network agent with multiple heads for parallel environments.

    This agent implements a DQN algorithm adapted for environments with multiple parallel agents.
    It maintains a Q-network and a target network for value estimation, using experience replay
    for training.

    Args:
        env (ParallelEnv): A parallel environment interface that provides observation and action spaces
                           for multiple agents.
        args (Args, optional): Configuration parameters for the agent. If None, default Args() will be used.

    Attributes:
        args (Args): Configuration parameters for the agent.
        device (torch.device): Device to run computations on (CPU/GPU).
        env (ParallelEnv): Reference to the parallel environment.
        obs_size (int): Size of the flattened observation space.
        actions_n (list): List of action space sizes for each agent.
        q_network (MultiHeadQNetwork): Main Q-network for action value estimation.
        target_network (MultiHeadQNetwork): Target network for stable learning.
        optimizer (torch.optim.Adam): Optimizer for Q-network parameters.
        rb (ReplayBuffer): Buffer storing experience transitions.
        rb_idx (int): Current index in the replay buffer.
        trained (bool): Flag indicating if the agent has been trained.

    Methods:
        train: Main training loop for the agent.
        train_step: Performs a single optimization step.
        act_train: Selects actions during training (with exploration).
        act: Selects actions using the current policy (without exploration).
        store: Stores transitions in the replay buffer.
        save_agent: Saves the Q-network parameters to disk.
        load_agent: Loads Q-network parameters from disk.
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

        # env setup
        self.env = env
        self.get_spaces()
        torch_compile = False

        self.q_network = MultiHeadQNetwork(
            obs_size=self.obs_size,
            actions_n=self.actions_n,
            device=self.device,
            torch_compile=torch_compile,
        )
        self.target_network = MultiHeadQNetwork(
            obs_size=self.obs_size,
            actions_n=self.actions_n,
            device=self.device,
            torch_compile=torch_compile,
        )

        self.optimizer = optim.Adam(  # type: ignore
            self.q_network.parameters(), lr=self.args.learning_rate
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            capacity=self.args.buffer_size, seed=self.args.seed, device=self.device
        )
        self.rb_idx = 0

    def get_spaces(self):
        """Get observation and action spaces from environment.

        This method processes the environment's observation and action spaces to set up:
        1. The total observation size (self.obs_size) by summing up each agent's observation space dimension
        2. The number of actions available (self.actions_n) for each agent

        The method assumes observation spaces are continuous with shape attribute and action spaces are discrete with n attribute.

        Attributes modified:
            self.obs_size (int): Total size of observation space across all agents
            self.actions_n (list): List containing number of actions available for each agent
        """
        self.obs_size = 0
        for agent in self.env.observation_spaces:
            obs_space = self.env.observation_spaces[agent]
            self.obs_size += obs_space.shape[0]  # type: ignore

        self.actions_n = []
        for agent in self.env.action_spaces:
            act_space = self.env.action_spaces[agent]
            self.actions_n.append(act_space.n)  # type: ignore

    def seed(self, seed):
        """
        Seeds all random number generators for reproducibility.

        Args:
            seed (int): The seed value to use for random number generation.
                        The same seed will be used for Python's random module,
                        NumPy's random number generator, PyTorch's random number
                        generator, and PyTorch's CUDNN backend.

        Notes:
            - This method ensures deterministic behavior when the same seed is used
            - Sets PyTorch's CUDNN backend to deterministic mode based on args.torch_deterministic
        """
        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def train(
        self,
        env: ParallelEnv,
        log_path: Path | str | None = None,
        run_name: str | None = None,
    ):
        """Train the agent in the given environment.

        This method implements the main training loop for the DQN agent. It handles the interaction
        between the agent and environment, collecting experiences, training the network, and logging
        metrics.

        Args:
            env (ParallelEnv): A parallel environment instance that implements the PettingZoo ParallelEnv interface
            log_path (Path | str | None, optional): Path where to save the training logs. Defaults to "runs".
            run_name (str | None, optional): Name for the training run. If None, generates one using timestamp.

        Returns:
            None: The method updates the agent's internal state and saves training artifacts.

        Details:
            - Creates directories for logging and saving models
            - Saves hyperparameters and training configuration
            - Implements the main training loop:
                * Collects experiences through environment interaction
                * Stores transitions in replay buffer
                * Performs DQN training updates
                * Logs metrics and episode returns
                * Saves model checkpoints periodically
            - Uses tensorboard for metric visualization
            - Saves complete environment traces if configured
        """
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

        # Start the game
        obs, infos = env.reset()
        episodic_returns = [0] * len(infos)
        if self.args.save_all_trace:
            trace = Trace()
        else:
            trace = RewardTrace()
        episode = 0
        all_done = [False] * len(infos)

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if any(all_done):
                obs, infos = env.reset()
                all_done = [False] * len(infos)
                episodic_returns = [0] * len(infos)
                trace.empty()
                episode += 1
            # ---------------------------

            # Get actions
            # ---------------------------
            actions = self.act_train(observation=obs, global_step=global_step)
            # ---------------------------

            # Perform action
            # ---------------------------
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            # ---------------------------

            # Save data and log
            # ---------------------------
            trace.add(
                actions=actions,
                observations=obs,
                rewards=rewards,
                terminations=terminated,
                truncations=truncated,
                infos=infos,
                episode=episode,
            )

            all_reward = []
            all_done = []
            for i, agent in enumerate(env.possible_agents):
                episodic_returns[i] += rewards[agent]  # type: ignore
                all_reward.append(rewards[agent])

                if terminated[agent] or truncated[agent]:
                    done = True
                else:
                    done = False
                all_done.append(done)
            actions = np.array([actions[agent] for agent in actions])
            flat_obs = np.concatenate([obs[agent] for agent in obs])
            self.store(
                state=flat_obs,
                action=actions,
                reward=np.array(all_reward),
                done=np.array(all_done),
            )

            if any(all_done):
                flat_next_obs = np.concatenate([next_obs[agent] for agent in obs])
                self.store(state=flat_next_obs)

                for i, agent_name in enumerate(rewards):
                    if self.args.verbose:
                        print(
                            f"global_step={global_step}, Agent: {agent_name} - Ep. return={episodic_returns[i]}"
                        )
                        print(
                            f"global_step={global_step}, Total Ep. return={np.sum(episodic_returns)}"
                        )
                    writer.add_scalar(
                        f"returns/{agent_name}",
                        episodic_returns[i],
                        global_step,
                    )
                    writer.add_scalar(
                        "returns/total",
                        np.sum(episodic_returns),
                        global_step,
                    )
                if hasattr(trace, "add_final_obs"):
                    trace.add_final_obs(next_obs)
                trace.save_trace(save_path=save_path, episode=episode)

            if global_step % 100000 == 0 and self.args.save_model:
                with open(save_path / "last_model.json", "w") as f:
                    json.dump({"save_step": global_step}, f)
                self.save_agent(save_path)
            # ---------------------------

            obs = next_obs

            # ALGO LOGIC: training.
            # ================================
            if global_step > self.args.learning_starts:
                self.train_step(global_step=global_step, writer=writer)
            # ================================

        if self.args.save_model:
            with open(save_path / "last_model.json", "w") as f:
                json.dump({"save_step": global_step}, f)  # type: ignore
            self.save_agent(save_path=save_path)

        writer.close()
        self.trained = True

    def train_step(self, global_step: int, writer: None | SummaryWriter):
        """Train the DQN agent for one step.

        This method performs a single training step of the DQN algorithm, including:
        1. Computing TD targets and loss
        2. Updating the Q-network parameters
        3. Periodically updating the target network
        4. Logging metrics if a writer is provided

        Args:
            global_step (int): The current global step count in training
            writer (None | SummaryWriter): TensorBoard writer for logging. If None, no logging is performed

        Returns:
            None

        Note:
            - Training occurs every `train_frequency` steps
            - Target network updates every `target_network_frequency` steps
            - Uses MSE loss between current Q-values and TD targets
            - Implements soft update for target network using parameter `tau`
        """
        if global_step % self.args.train_frequency == 0:
            # Get Loss
            # ---------------------------
            data, _ = self.rb.sample(self.args.batch_size)
            with torch.no_grad():
                target_max, _ = self.target_network(data.next_observations).max(dim=-1)
                td_target = data.rewards + self.args.gamma * target_max * (
                    1 - data.dones
                )
            old_val = (
                self.q_network(data.observations)
                .gather(-1, data.actions.unsqueeze(-1).to(torch.int64))
                .squeeze(-1)
            )

            loss = F.mse_loss(target=td_target, input=old_val)
            # ---------------------------

            # Log
            # ---------------------------
            if global_step % 100 == 0 and writer is not None:
                writer.add_scalar("losses/mh_dqn_td_loss", loss, global_step)
                writer.add_scalar(
                    "losses/mh_dqn_q_values", old_val.mean().item(), global_step
                )
            # ---------------------------

            # optimize the model
            # ---------------------------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ---------------------------

        # update target network
        # ---------------------------
        if global_step % self.args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_network_param.data.copy_(
                    self.args.tau * q_network_param.data
                    + (1.0 - self.args.tau) * target_network_param.data
                )
        # ---------------------------

    def save_agent(self, save_path: Path | str, name: None | str = None):
        """Save the agent's Q-network to a file.

        This method saves the state dictionary of the Q-network to a specified path. It creates
        the necessary directories if they don't exist and verifies the save operation by attempting
        to load the saved model.

        Args:
            save_path (Union[Path, str]): The base directory path where the model will be saved.
                A 'models' subdirectory will be created within this path.
            name (Optional[str]): The name to use for the saved model file. If None,
                defaults to "trained_model". The .pth extension will be added automatically.

        Returns:
            None

        Examples:
            >>> agent.save_agent('/path/to/save', 'my_model')
            # Saves to /path/to/save/models/my_model.pth

        Note:
            The method will print a success message if the save operation is verified,
            or an error message if the save operation could not be verified.
        """
        save_path = Path(save_path)
        if name is None:
            name = "trained_model"
        model_save_path = save_path / "models" / f"{name}.pth"
        if not model_save_path.parent.exists():
            os.makedirs(model_save_path.parent)
        torch.save(self.q_network.state_dict(), model_save_path)
        if self.load_agent(save_path, name=f"{name}.pth"):
            print(f"model saved to {model_save_path}")
        else:
            print("Could not save the model!")

    def act_train(
        self,
        observation: Dict[str, np.ndarray],
        global_step: int,
    ) -> dict:
        """
        Get actions during training phase with epsilon-greedy policy.

        This method implements epsilon-greedy action selection where random actions are taken with
        probability epsilon, which decays linearly from start_e to end_e over the course of training.
        Otherwise, actions are selected using the trained policy.

        Args:
            observation (Dict[str, np.ndarray]): Current observation from the environment for each agent
            global_step (int): Current global timestep in the training process

        Returns:
            dict: Dictionary mapping agent IDs to their selected actions
        """
        epsilon = linear_schedule(
            self.args.start_e,
            self.args.end_e,
            int(self.args.exploration_fraction * self.args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = {
                agent: self.env.action_space(agent).sample()
                for agent in self.env.agents
            }
        else:
            actions = self.act(observation=observation, no_grad=True)
        return actions

    def act(self, observation: dict, no_grad: bool = True) -> dict:
        """Takes a batched observation and returns an action for each agent.

        The method processes the observations from multiple agents, passes them through the Q-network,
        and converts the Q-values into actions. Can be executed with or without gradient computation.

        Args:
            observation (dict): Dictionary containing observations for each agent
                               with agent IDs as keys and observations as values.
            no_grad (bool, optional): If True, disables gradient computation. Defaults to True.

        Returns:
            dict: Dictionary mapping agent IDs to their corresponding actions.
        """
        flat_obs = np.concatenate([observation[agent] for agent in observation])
        if no_grad:
            with torch.no_grad():
                q_values = self.q_network(
                    torch.Tensor(np.array([flat_obs])).to(self.device)
                )
        else:
            q_values = self.q_network(
                torch.Tensor(np.array([flat_obs])).to(self.device)
            )
        actions = self.qval_to_act(q_values=q_values, agents=list(observation.keys()))
        return actions

    def qval_to_act(self, q_values: torch.Tensor, agents: list) -> dict:
        """Convert Q-values to action dictionary.

        This method takes the Q-values tensor output from the network and converts it to a dictionary
        mapping agent names to their corresponding selected actions.

        Args:
            q_values (torch.Tensor): Tensor of shape (batch_size, num_agents, num_actions) containing
                the Q-values for each agent and action.
            agents (list): List of agent names/IDs.

        Returns:
            dict: Dictionary mapping agent names to their selected actions (indices of max Q-values).
        """
        actions = torch.argmax(q_values, dim=-1).cpu().numpy()[0]
        actions_dict = {agent: actions[i] for i, agent in enumerate(agents)}
        return actions_dict

    def load_agent(
        self, load_path: Path | str, name: str = "trained_model.pth"
    ) -> bool:
        """Loads the agent's Q-network model from a specified path.

        This method attempts to load a pre-trained model state dictionary into the agent's Q-network
        from a given filesystem path.

        Args:
            load_path (Union[Path, str]): Base directory path where the model file is stored.
            name (str, optional): Name of the model file. Defaults to "trained_model.pth".

        Returns:
            bool: True if model was successfully loaded, False otherwise.

        Raises:
            None: Exceptions during loading are caught and handled internally.

        Note:
            The actual model path is constructed by joining the base `load_path` with "models"
            directory and the model filename.
        """
        load_path = Path(load_path) / "models" / name
        if load_path.exists():
            try:
                self.q_network.load_state_dict(torch.load(load_path))
                return True
            except Exception as e:
                print("#######################")
                print(f"Could not load the model from {load_path}")
                print(e)
                print("#######################")
                return False
        else:
            print("#######################")
            print(f"Path {load_path} does not exist.")
            print("#######################")
            return False

    def store(self, state, action=None, reward=None, done=None):
        """Store a transition (state, action, reward, done) into the replay buffer.

        Args:
            state: The current state
            action (optional): The action taken in the current state
            reward (optional): The reward received
            done (optional): Whether the episode ended after this transition

        Note:
            The method updates the replay buffer index (rb_idx) after storing the transition.
        """
        self.rb.push(
            state=state, idx=self.rb_idx, action=action, reward=reward, done=done
        )
        self.rb_idx += 1
