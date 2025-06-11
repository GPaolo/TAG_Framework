import os
import random
import time
import itertools
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from dataclasses_json import dataclass_json
from tame.data_handling.trace import Trace, RewardTrace
from tame.agents.base_dqn import DQN, linear_schedule
import json
from dataclasses import dataclass
from tame.hierarchy.base_agent import LevelAgent
from typing import Dict
from gymnasium.spaces import Box, Discrete
import gymnasium
from pettingzoo import ParallelEnv
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


class Agent(LevelAgent):
    """A Deep Q-Network (DQN) agent implementation for multi-agent environments.

    It wraps around a basic DQN implementation to treat a multi-agent environment as a single-agent environment by concatenating all
    observations and creating a mapping between individual agent actions and a single action space.

    Args:
        observation_space (gymnasium.spaces.Dict): The observation space dictionary for all agents.
        action_space (gymnasium.spaces.Dict): The action space dictionary for all agents.
        device (torch.device): The device (CPU/GPU) to run the agent on.
        name (str): Name identifier for the agent. Defaults to "monolithic_dqn".
        args (Args | None): Configuration arguments for the agent. Defaults to None.
        torch_compile (bool): Whether to compile the model using torch.compile(). Defaults to False.

    Attributes:
        obs_size (int): Size of the flattened observation space.
        actions_n (int): Total number of possible combined actions.
        index_to_actions (dict): Mapping from action indices to actual action combinations.
        actions_to_index (dict): Mapping from action combinations to action indices.
        agent (DQN): The underlying DQN agent that handles the learning process.

    Methods:
        train(env, log_path=None, run_name=None): Train the agent in the given environment.
        act_train(observation, global_step): Select actions during training (with exploration).
        act(observation): Select actions using the trained policy.
        update_step(global_step, writer): Perform a training update step.
        save_agent(save_path, name=None): Save the agent's model to disk.
        load_agent(load_path, name="trained_model.pth"): Load the agent's model from disk.
        store(state, action=None, reward=None, done=None): Store a transition in the agent's replay buffer.

    Notes:
        As it respects the LevelAgent interface, it can be used in hierarchical learning setups.
        It can also be used as a standalone agent.
        This is the DQN subagent that we used in all our hierarchical agents.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        action_space: gymnasium.spaces.Dict,
        device: torch.device,
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
        self.get_spaces()
        self.device = device
        self.name = name

        self.agent = DQN(
            observation_space=Box(np.inf, np.inf, shape=[self.obs_size]),
            action_space=Discrete(self.actions_n),
            args=self.args,  # type: ignore
            torch_compile=torch_compile,
            device=self.device,
            name="dqn",
        )
        self.agent.seed(seed=self.args.seed)

    def get_spaces(self):
        """Calculate observation and action spaces dimensions for the DQN agent.

        This method processes observation and action spaces from multiple agents to create:
        1. Combined observation space size by summing individual observation dimensions
        2. Combined action space size by multiplying individual action space sizes
        3. Mappings between flattened action indices and actual action combinations

        The method updates the following instance attributes:
        - obs_size: Combined size of observation space across all agents
        - actions_n: Total number of possible action combinations
        - index_to_actions: Dictionary mapping indices to action combinations
        - actions_to_index: Dictionary mapping action combinations to indices

        Note:
            Assumes observation spaces are Box spaces with shape attribute
            Assumes action spaces are Discrete spaces with n attribute

        Returns:
            None
        """
        self.obs_size = 0
        for agent in self.observation_space:
            obs_space = self.observation_space[agent]
            self.obs_size += obs_space.shape[0]  # type: ignore

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
        """Sets all random seeds for reproducibility.

        This method initializes random seeds for Python's random module, NumPy, PyTorch,
        and configures CUDNN deterministic behavior to ensure reproducible results across runs.

        Args:
            seed (int): The random seed value to use for all random number generators.
        """
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
        """Train the DQN agent in a given environment.

        This method implements the training loop for a DQN agent, handling episode management,
        action selection, experience storage, and model updates. It also manages logging and
        model checkpointing.

        Args:
            env (ParallelEnv): The environment to train in. Must implement the pettingzoo ParallelEnv interface.
            log_path (Path | str | None, optional): Directory path for saving logs and models.
                Defaults to "runs" if None.
            run_name (str | None, optional): Name for the training run. If None, generates a name
                using experiment name, seed and timestamp.

        Returns:
            None

        Side Effects:
            - Creates log directories and saves parameter configurations
            - Writes training metrics to TensorBoard
            - Saves model checkpoints if save_model=True
            - Sets self.trained to True upon completion
            - Closes the TensorBoard writer

        Notes:
            The training process includes:
            - Episode management with environment resets
            - Action selection and environment stepping
            - Experience storage in replay buffer
            - Periodic model updates after learning_starts steps
            - Logging of returns and other metrics
            - Model checkpointing every 100000 steps if enabled
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

            all_reward = np.array([np.sum(all_reward)])
            self.store(
                state=flat_obs,
                action=flat_actions,
                reward=all_reward,
                done=all_done,
            )

            if all_done:
                flat_next_obs = np.concatenate([next_obs[agent] for agent in obs])
                self.store(state=flat_next_obs)

                for i, agent_name in enumerate(rewards):
                    if self.args.verbose:
                        print(
                            f"global_step={global_step}, Agent: {agent_name} - Ep. return={episodic_returns[i]}"
                        )
                    writer.add_scalar(
                        f"returns/{agent_name}",
                        episodic_returns[i],
                        global_step,
                    )
                if self.args.verbose:
                    print(
                        f"global_step={global_step}, Total Ep. return={np.sum(episodic_returns)}"
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
            if global_step > self.args.learning_starts:
                self.update_step(global_step=global_step, writer=writer)
            # ================================

        if self.args.save_model:
            with open(save_path / "last_model.json", "w") as f:
                json.dump({"save_step": global_step}, f)  # type: ignore
            self.save_agent(save_path=save_path)

        writer.close()
        self.trained = True

    def update_step(self, global_step: int, writer: None | SummaryWriter):
        """
        Update the agent's learning process for a single step.

        Args:
            global_step (int): The current global step count in the training process.
            writer (None | SummaryWriter): TensorBoard SummaryWriter instance for logging training metrics.
                                          If None, no logging will be performed.

        Returns:
            None
        """
        self.agent.update_step(global_step=global_step, writer=writer)

    def save_agent(self, save_path: Path | str, name: None | str = None):
        """Save the agent's model to disk.

        This method saves the state dictionary of the Q-network to a specified path.
        After saving, it attempts to load the model to verify the save was successful.

        Args:
            save_path (Union[Path, str]): Directory path where to save the model.
            name (Optional[str]): Name of the model file. Defaults to "trained_model".

        Returns:
            None

        Example:
            >>> agent.save_agent('/path/to/save', 'my_model')
            # Saves model to '/path/to/save/models/my_model.pth'
        """
        save_path = Path(save_path)
        if name is None:
            name = "trained_model"
        model_save_path = save_path / "models" / f"{name}.pth"
        if not model_save_path.parent.exists():
            os.makedirs(model_save_path.parent)
        torch.save(self.agent.q_net.state_dict(), model_save_path)
        if self.load_agent(save_path, name=f"{name}.pth"):
            print(f"model saved to {model_save_path}")
        else:
            print("Could not save the model!")

    def act_train(
        self,
        observation: Dict[str, np.ndarray],
        global_step: int,
    ) -> Dict[str, np.ndarray]:
        """Takes an observation and global step, returns actions using epsilon-greedy exploration.

        This method implements epsilon-greedy exploration where the agent either takes a random action
        with probability epsilon, or the predicted best action according to the current policy.
        Epsilon decays linearly over time from start_e to end_e.

        Args:
            observation (Dict[str, np.ndarray]): Current observation from the environment
            global_step (int): Current global timestep used for epsilon decay scheduling

        Returns:
            Dict[str, np.ndarray]: Selected action (either random or from policy)
        """
        epsilon = linear_schedule(
            self.args.start_e,
            self.args.end_e,
            int(self.args.exploration_fraction * self.args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = self.action_space.sample()
        else:
            actions = self.act(observation=observation)
        return actions

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Selects actions for agents based on the given observation.

        Args:
            observation (Dict[str, np.ndarray]): A dictionary mapping agent IDs to their observations,
                where each observation is a numpy array.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping agent IDs to their selected actions,
                where each action is a numpy array.

        Notes:
            The method flattens the observations from all agents into a single array,
            passes it through the DQN agent to get actions indices, converts indices
            to actual actions using index_to_actions mapping, and returns a dictionary
            with actions for each agent.
        """
        flat_obs = np.concatenate([observation[agent] for agent in observation])
        actions = self.agent.act(flat_obs)
        actions = self.index_to_actions[actions]
        actions = {agent: action for agent, action in zip(observation, actions)}
        return actions

    def load_agent(
        self, load_path: Path | str, name: str = "trained_model.pth"
    ) -> bool:
        """Loads a trained model from disk.

        Args:
            load_path (Union[Path, str]): Directory path where the model is stored.
            name (str, optional): Name of the model file. Defaults to "trained_model.pth".

        Returns:
            bool: True if model was successfully loaded, False otherwise.

        Notes:
            The model file is expected to be in a 'models' subdirectory under the provided load_path.
            The function attempts to load the model state dictionary into the agent's Q-network.
        """
        load_path = Path(load_path) / "models" / name
        if load_path.exists():
            try:
                self.agent.q_net.load_state_dict(torch.load(load_path))
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
        """
        Store a transition in the replay buffer.

        Args:
            state: Current state observation
            action (optional): Action taken in current state
            reward (optional): Reward received after taking action
            done (optional): Boolean indicating if episode ended after action

        The method stores the (state, action, reward, done) transition tuple in the agent's replay buffer
        for later training. The action, reward and done parameters are optional to allow storing
        just the initial state.
        """
        self.agent.store(
            state=state,
            action=action,
            reward=reward,
            done=done,
        )
