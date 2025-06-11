import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium
import json
import time
import itertools
import random
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from tame.hierarchy.base_agent import LevelAgent
from tame.agents.base_ppo import PPO
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os
from functools import cached_property
from gymnasium.spaces import Box, Discrete
from tame.data_handling.trace import Trace, RewardTrace
from tame.utils.utils import filter_unexpected_fields
import torch.nn as nn
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
    learn_comm: bool = False
    ae_epochs: int = 50

    @cached_property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class Autoencoder(nn.Module):
    """Autoencoder neural network module use as learned communication function phi.

    This class implements a simple autoencoder architecture with one hidden layer
    for both encoder and decoder. The encoder compresses the input into a lower
    dimensional latent space, while the decoder attempts to reconstruct the original
    input from the latent representation.

    Args:
        input_dim (int): Dimension of the input data
        feature_dim (int): Dimension of the latent space (compressed representation)
        hidden_dim1 (int, optional): Number of neurons in hidden layer. Defaults to 32.

    Attributes:
        encoder (nn.Sequential): Neural network for encoding input to latent space
        decoder (nn.Sequential): Neural network for decoding latent space to reconstruction
        criterion (nn.MSELoss): Mean squared error loss function for reconstruction

    Methods:
        forward(x): Returns encoded representation in latent space
        full_forward(x): Returns full reconstruction of input data
    """

    def __init__(self, input_dim: int, feature_dim: int, hidden_dim1: int = 32):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, feature_dim),
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor to be encoded.

        Returns:
            encoded: Encoded representation of the input tensor after passing through the encoder network.
        """
        # Encode
        encoded = self.encoder(x)
        return encoded

    def full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a full forward pass through the encoder-decoder architecture.

        This method takes an input tensor, passes it through the encoder to obtain
        a latent representation, and then decodes it back to the original space.

        Args:
            x (torch.Tensor): Input tensor to be encoded and decoded.

        Returns:
            torch.Tensor: Reconstructed tensor after encoding and decoding.
        """
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded


class Agent(LevelAgent):
    """A monolithic PPO agent that can handle multi-agent environments by flattening observations and actions.

    It wraps around a basic PPO implementation to treat a multi-agent environment as a single-agent environment by concatenating all
    observations and creating a mapping between individual agent actions and a single action space.
    It implements the Proximal Policy Optimization (PPO) algorithm for training.

    Args:
        observation_space (gymnasium.spaces.Dict): Dictionary of observation spaces for each agent.
        action_space (gymnasium.spaces.Dict): Dictionary of action spaces for each agent.
        device (torch.device): Device to run the model on (CPU or GPU).
        communication_space (gymnasium.spaces.Dict | None): Optional dictionary of communication spaces for hierarchical learning.
        name (str): Name identifier for the agent. Defaults to "monolithic_dqn".
        args (Args | None): Configuration arguments for the agent. If None, default Args will be used.
        torch_compile (bool): Whether to compile the model using torch.compile(). Defaults to False.

    Attributes:
        args (Args): Configuration arguments for the agent.
        observation_space (gymnasium.spaces.Dict): Dictionary of observation spaces.
        action_space (gymnasium.spaces.Dict): Dictionary of action spaces.
        communication_space (gymnasium.spaces.Dict | None): Dictionary of communication spaces.
        device (torch.device): Device where the model runs.
        name (str): Agent identifier.
        agent (PPO): The underlying PPO agent.
        obs_size (int): Size of flattened observation space.
        actions_n (int): Size of flattened action space.
        index_to_actions (dict): Mapping from flattened action index to agent actions.
        actions_to_index (dict): Mapping from agent actions to flattened action index.
        phi (Autoencoder | None): Optional autoencoder for communication learning.

    Methods:
        act: Select actions using the trained policy.
        act_train: Select actions during training (with exploration).
        update_step: Perform a training update step.
        comm: Generate communication output for hierarchical learning.
        store: Store transitions in the replay buffer.
        train: Train the agent in an environment.
        save_agent: Save the agent's model.
        load_agent: Load a saved model.

    Notes:
        As it respects the LevelAgent interface, it can be used in hierarchical learning setups.
        It can also be used as a standalone agent.
        This is the PPO subagent that we used in all our hierarchical agents.
        Moreover it can only handle discrete action spaces.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        action_space: gymnasium.spaces.Dict,
        device: torch.device,
        communication_space: gymnasium.spaces.Dict | None = None,
        name: str = "monolithic_dqn",
        args: None | Args = None,
        torch_compile: bool = False,
    ) -> None:
        if args is None:
            self.args: Args = Args()
        else:
            self.args: Args = args

        self.seed(self.args.seed)

        # env setup
        self.observation_space = observation_space
        self.action_space = action_space
        self.communication_space = communication_space

        self.get_spaces()
        self.device = device
        self.name = name

        self.agent = PPO(
            observation_space=Box(np.inf, np.inf, shape=[self.obs_size]),
            action_space=Discrete(self.actions_n),
            args=self.args,  # type: ignore
            torch_compile=torch_compile,
            device=self.device,
            name=self.name,
        )
        self.agent.seed(seed=self.args.seed)

        if hasattr(self.args, "learn_comm") and self.args.learn_comm:
            self.phi = None

    def get_spaces(self):
        """
        Processes observation, action, and communication spaces to set up necessary instance variables.

        This method initializes:
        - obs_size: Total size of observation space across all agents
        - comm_output_size: Total size of communication space across all agents (if communication enabled)
        - actions_n: Total number of possible action combinations
        - index_to_actions: Mapping from action indices to actual action combinations
        - actions_to_index: Mapping from action combinations to indices

        The method handles both single and multi-agent scenarios by:
        1. Summing observation dimensions across agents
        2. Summing communication dimensions if enabled
        3. Computing all possible action combinations and creating bidirectional mappings

        Note:
            Assumes observation_space and action_space are dictionary spaces with agents as keys.
            For action spaces, assumes they are discrete spaces with property 'n'.
        """
        self.obs_size = 0
        for agent in self.observation_space:
            obs_space = self.observation_space[agent]
            self.obs_size += obs_space.shape[0]  # type: ignore

        if self.communication_space is not None:
            self.comm_output_size = 0
            for agent in self.communication_space:
                comm_space = self.communication_space[agent]
                self.comm_output_size += comm_space.shape[0]  # type: ignore

        self.actions_n = 1
        possible_values = []
        for agent in self.action_space:
            act_space = self.action_space[agent]
            self.actions_n *= act_space.n  # type: ignore
            possible_values.append(list(range(act_space.n)))  # type: ignore

        combinations = list(itertools.product(*possible_values))
        self.index_to_actions = {
            i: combination for i, combination in enumerate(combinations)
        }
        self.actions_to_index = {
            combination: i for i, combination in enumerate(combinations)
        }

    def seed(self, seed):
        """Sets random seeds for reproducibility.

        This method initializes random number generators with a specified seed value to ensure
        reproducible results across different runs of the code.

        Args:
            seed (int): The seed value used to initialize random number generators.
                The same seed will produce the same sequence of random numbers.

        Notes:
            - Sets seed for Python's random module
            - Sets seed for NumPy's random number generator
            - Sets seed for PyTorch's random number generator
            - Configures PyTorch's CUDNN backend determinism based on args setting
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Selects actions for each agent based on the given observation.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary containing observations for each agent
                with agent IDs as keys and observation arrays as values.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing selected actions for each agent
                with agent IDs as keys and action arrays as values.

        Note:
            - Flattens input observations into a single array
            - Uses internal agent model to select action index
            - Converts index to actual action using lookup table
            - Maps actions back to individual agents
        """
        flat_obs = np.concatenate(list(observation.values()))
        actions = self.agent.act(flat_obs)
        actions = self.index_to_actions[int(actions)]
        actions = {agent: action for agent, action in zip(observation, actions)}
        return actions

    def act_train(
        self, observation: Dict[str, np.ndarray], global_step: int
    ) -> Dict[str, np.ndarray]:
        """
        Selects actions during training phase based on the current observation.

        This method processes observations through the agent's policy network and returns
        corresponding actions for each agent in the environment.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary containing observations for each agent
                where keys are agent identifiers and values are numpy arrays of observations.
            global_step (int): Current training step count used for tracking training progress.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping agent identifiers to their corresponding
                selected actions as numpy arrays.
        """
        flat_obs = np.concatenate(list(observation.values()))
        actions = self.agent.act_train(flat_obs, global_step=global_step)
        actions = self.index_to_actions[int(actions)]
        actions = {agent: action for agent, action in zip(observation, actions)}
        return actions

    def update_step(self, global_step: int, writer: None | SummaryWriter):
        """
        Updates the agent and communication function if applicable.

        This method checks if communication learning is enabled and, if so, updates
        communication function through the update_phi method before proceeding with
        the main agent update logic. It optionally logs training progress and metrics.

        Args:
            global_step (int): The current training step, used for logging and
                controlling training schedule.
            writer (SummaryWriter | None): A writer for logging training metrics. If
                None, logging is skipped.

        Returns:
            None
        """
        if (
            hasattr(self.args, "learn_comm")
            and self.args.learn_comm
            and self.communication_space is not None
        ):
            self.update_phi(global_step=global_step, writer=writer)

        self.agent.update_step(global_step=global_step, writer=writer)

    def update_phi(self, global_step: int, writer: None | SummaryWriter):
        """Updates the communication function by training the autoencoder with data collected in
        the agent's state buffer. Once the buffer is filled, this method retrieves a subset of
        the observations, constructs a DataLoader, and performs a series of training epochs on
        the autoencoder.

        Args:
            global_step (int): The current training step, used for logging.
            writer (Optional[SummaryWriter]): SummaryWriter for TensorBoard logging. If provided,
                logs the average reconstruction loss after each epoch.

        Raises:
            ValueError: If the communication function (phi) is not initialized before calling this method."""

        # We update with the same frequency as the agent...
        assert self.phi is not None, ValueError(
            f"{self.name}: phi is not defined yet None. Call comm first before updating phi"
        )

        if self.agent.buffer_idx >= self.agent.buffer_size:
            # Just take obs from l-1
            # TODO improve this by saving just the obs. As this can get messy if order is not right
            dataset = self.agent.states[:, : self.comm_input_size].to(self.device)
            train_loader = torch.utils.data.DataLoader(  # type: ignore
                dataset, batch_size=256, shuffle=True
            )

            for epoch in range(self.args.ae_epochs):
                total_loss = 0
                for data in train_loader:
                    # Assuming data is already flattened
                    inputs = data.view(data.size(0), -1)

                    # Forward pass
                    outputs = self.phi.full_forward(inputs)
                    loss = self.phi.criterion(outputs, inputs)

                    # Backward pass and optimization
                    self.comm_optimizer.zero_grad()
                    loss.backward()
                    self.comm_optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                if writer is not None:
                    writer.add_scalar(
                        f"{self.name}/ae_loss",
                        avg_loss,
                        global_step,
                    )

    def init_phi(self, input_size):
        """
        Initializes the phi communication function.

        This method:
        - Stores the provided input size for future use.
        - Instantiates an Autoencoder-based phi communication function with the specified dimensions.
        - Creates and configures an optimizer to update the Autoencoder parameters.

        Args:
            input_size (int): The dimension of the input for the feature extractor.

        Returns:
            torch.nn.Module: The initialized Autoencoder model.
        """
        self.comm_input_size = input_size
        # Instantiate phi here so no worries about the dimension shapes
        self.phi = Autoencoder(
            input_dim=self.comm_input_size, feature_dim=self.comm_output_size
        ).to(self.device)
        self.comm_optimizer = torch.optim.Adam(  # type: ignore
            self.phi.parameters(), lr=self.args.learning_rate
        )
        return self.phi

    def comm(self, observation: np.ndarray | Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generates a communication representation of the agent’s state to be sent to a higher-level agent.

        If the agent is configured to learn communication (i.e., has a communication space and
        args.learn_comm is set), this method passes the observation through a learned model to compute
        the communication vector. Otherwise, it returns the observation directly.

        Args:
            observation (np.ndarray | Dict[str, np.ndarray]): The agent’s observation, which may be
                a NumPy array or a dictionary of NumPy arrays.

        Returns:
            np.ndarray: The resulting communication vector. If communication is learned, this is the output
            of the learned model. If not, this is simply the original observation.

        Raises:
            ValueError: If the agent does not have a communication space defined.
        """
        if self.communication_space is None:
            raise ValueError(
                f"Agent {self.name} has not communication space defined so it cannot generate comm."
            )
        if hasattr(self.args, "learn_comm") and self.args.learn_comm:
            if isinstance(observation, dict):
                observation = np.concat([obs for obs in observation.values()])

            if self.phi is None:
                self.phi = self.init_phi(input_size=len(observation))

            observation = torch.Tensor(observation).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                comm = self.phi(observation)
            comm = comm.detach().cpu().numpy()[0]  # [0] to remove the batch dimension
            return comm
        else:
            if isinstance(observation, dict):
                observation = np.concat([obs for obs in observation.values()])
            return observation

    def save_agent(self, save_path: Path | str, name: None | str = None):
        """
        Saves the agent's model to a specified directory path. If no name is provided,
        the model file defaults to "trained_model.pth".

        Args:
            save_path (Path | str): The path where the model will be saved.
            name (None | str, optional): An optional custom name for the saved model file.
                Defaults to "trained_model".

        Returns:
            None
                Prints a message indicating if the model was successfully saved.
        """
        save_path = Path(save_path)
        if name is None:
            name = "trained_model"
        model_save_path = save_path / "models" / f"{name}.pth"
        if not model_save_path.parent.exists():
            os.makedirs(model_save_path.parent)
        torch.save(self.agent.actor_critic.state_dict(), model_save_path)
        if self.load_agent(save_path, name=name):
            print(f"{self.name}: model saved to {model_save_path}")
        else:
            print(f"{self.name}: Could not save the model!")

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """
        Load a trained policy from disk and update the agent's actor-critic model parameters.

        This method constructs the full path to the model file by appending "models/<name>.pth"
        to the specified load_path. If the file is found and successfully loaded,
        the agent's state dictionary will be set accordingly. Otherwise, a warning is printed
        and the method returns False.

        Args:
            load_path (Path | str): The path to the directory containing the model checkpoint.
            name (str, optional): The filename (minus extension) of the checkpoint. Defaults to "trained_model".

        Returns:
            bool: True if the model was successfully loaded and applied; False otherwise.
        """
        load_path = Path(load_path) / "models" / f"{name}.pth"
        if load_path.exists():
            try:
                self.agent.actor_critic.load_state_dict(torch.load(load_path))
                return True
            except Exception as e:
                print(f"{self.name}: #######################")
                print(f"{self.name}: Could not load the model from {load_path}")
                print(f"{self.name}: {e}")
                print(f"{self.name}: #######################")
                return False
        else:
            print(f"{self.name}: #######################")
            print(f"{self.name}: Path {load_path} does not exist.")
            print(f"{self.name}: #######################")
            return False

    def store(
        self,
        state: dict | np.ndarray | torch.Tensor,
        action: dict | None | np.ndarray | torch.Tensor = None,
        reward: float | None = None,
        done: bool | None = None,
    ):
        """
        Stores an experience tuple to the agent's memory.

        This method processes the input state and action before storing the experience.
        If the state is provided as a dictionary, its values are concatenated into a single array.
        If the action is provided as a dictionary, it is converted to an index according to
        the mapping in `self.actions_to_index` and reshaped as a numpy array. The reward is
        converted to a float if provided.

        Args:
            state (dict | np.ndarray | torch.Tensor): The state observation. If a dictionary,
                the values will be concatenated.
            action (dict | None | np.ndarray | torch.Tensor, optional): The action taken.
                If a dictionary, it will be mapped to an index. Defaults to None.
            reward (float | None, optional): The reward received after taking the action.
                Defaults to None.
            done (bool | None, optional): A flag indicating if the episode has terminated.
                Defaults to None.

        Returns:
            None
        """
        if isinstance(state, dict):
            state = np.concatenate(list(state.values()))
        if action is not None and isinstance(action, dict):
            action = self.actions_to_index[tuple(action.values())]  # type: ignore
            action = np.atleast_1d(action)  # type: ignore
        if reward is not None:
            reward = float(reward)

        self.agent.store(
            state=state,
            action=action,  # type: ignore
            reward=reward,
            done=done,
        )

    def train(
        self, env: Any, log_path: Path | str | None = None, run_name: str | None = None
    ):
        """
        Trains the agent using the provided environment.

        This method initializes experiment parameters, sets up logging and tensorboard
        summary writer, and executes training over a specified number of timesteps. It
        handles environment resets, logging of hyperparameters, and optionally collects
        traces for reward analysis.

        Args:
            env (Any): The environment in which the agent is trained.
            log_path (Path | str | None, optional): The base directory for storing logs and
                training artifacts. Defaults to "runs" if None.
            run_name (str | None, optional): A unique identifier for the training run.
                If not provided, it is generated automatically based on the experiment name,
                seed, and current timestamp.

        Returns:
            None
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
        done = False
        episodic_returns = [0] * len(infos)
        if self.args.save_all_trace:
            trace = Trace()
        else:
            trace = RewardTrace()
        episode = 0
        all_done = False

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if all_done:
                obs, infos = env.reset()
                all_done = False
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
            flat_actions = np.array([self.actions_to_index[tuple(actions)]])
            flat_obs = np.concatenate([obs[agent] for agent in obs])
            all_done = any(all_done)

            all_reward = np.array(np.sum(all_reward))
            self.store(
                state=flat_obs,
                action=flat_actions,
                reward=all_reward,  # type: ignore
                done=all_done,
            )

            if all_done:
                flat_next_obs = np.concatenate([next_obs[agent] for agent in obs])
                self.store(state=flat_next_obs)

                for i, agent_name in enumerate(rewards):
                    if self.args.verbose:
                        print(
                            f"{self.name}: global_step={global_step}, Agent: {agent_name} - Ep. return={episodic_returns[i]}"
                        )
                    writer.add_scalar(
                        f"returns/{agent_name}",
                        episodic_returns[i],
                        global_step,
                    )
                if self.args.verbose:
                    print(
                        f"{self.name}: global_step={global_step}, Total Ep. return={np.sum(episodic_returns)}"
                    )
                writer.add_scalar(
                    "returns/total",
                    np.sum(episodic_returns),
                    global_step,
                )
                if hasattr(trace, "add_final_obs"):
                    trace.add_final_obs(next_obs)
                trace.save_trace(save_path=save_path, episode=str(episode))

            if global_step % 100000 == 0 and self.args.save_model:
                with open(save_path / "last_model.json", "w") as f:
                    json.dump({"save_step": global_step}, f)
                self.save_agent(save_path)
            # ---------------------------

            obs = next_obs

            # ALGO LOGIC: training.
            # ================================
            self.update_step(global_step=global_step, writer=writer)
            # ================================

        if self.args.save_model:
            with open(save_path / "last_model.json", "w") as f:
                json.dump({"save_step": global_step}, f)  # type: ignore
            self.save_agent(save_path=save_path)

        writer.close()
        self.trained = True
