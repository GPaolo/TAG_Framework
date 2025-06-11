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
from dataclasses import dataclass
from typing import List, Dict, Any
from tame.data_handling.trace import Trace, RewardTrace
import json
from scipy.optimize import linear_sum_assignment
from tame.hierarchy.base_agent import BaseAgent


@dataclass_json
@dataclass
class Args:
    """
    Configuration class for the training hyperparameters and settings.

    Attributes:
        exp_name (str): Name of the experiment, derived from the filename
        seed (int): Random seed for reproducibility
        torch_deterministic (bool): Whether to use deterministic algorithms in PyTorch
        verbose (bool): Enable/disable verbose output
        cuda (int): CUDA device ID for GPU acceleration
        save_model (bool): Whether to save the trained model
        save_all_trace (bool): Whether to save complete training traces
        total_timesteps (int): Total number of timesteps for training
        learning_rate (float): Learning rate for optimization
        buffer_size (int): Size of the replay buffer
        gamma (float): Discount factor for future rewards
        tau (float): Target network soft update rate
        target_network_frequency (int): Frequency of target network updates
        batch_size (int): Size of training batches
        start_e (float): Initial exploration rate
        end_e (float): Final exploration rate
        exploration_fraction (float): Fraction of total timesteps for exploration decay
        learning_starts (int): Number of timesteps before starting training
        train_frequency (int): Frequency of training updates
    """

    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
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


class GoalAssigner:
    """A class to assign goals to multiple agents in a coordinated manner.

    This class implements a high-level coordinator that assigns target goals to each subagent
    in a multi-agent system. It uses the Hungarian algorithm (linear sum assignment) to optimally
    match agents to goals based on distances.

    Args:
        n_subagent (int): Number of subagents/robots in the system.

    Methods:
        assign_goals(observation: Dict[str, np.ndarray]) -> dict:
            Assigns goals to each agent by minimizing the total distance between agents and goals.

            Args:
                observation (Dict[str, np.ndarray]): Dictionary containing observations for each agent.
                    Each observation should contain the agent's position and goal positions.

            Returns:
                dict: Dictionary mapping agent names to their assigned goal positions.
                    Format: {agent_name: np.ndarray([x, y])}

    Note:
        The observation dictionary for each agent should contain:
        - Agent position at indices [2:4]
        - Goal positions starting at index 4
    """

    def __init__(self, n_subagent: int) -> None:
        self.n_subagents = n_subagent

    def assign_goals(self, observation: Dict[str, np.ndarray]) -> dict:
        """Assigns goals to agents based on their positions using the Hungarian algorithm.

        This method takes the current observation of all agents and assigns goals to minimize
        the total distance between agents and their assigned goals.

        Args:
            observation (Dict[str, np.ndarray]): A dictionary containing observations for each agent.
                Each observation includes the agent's position and the relative positions of all goals.

        Returns:
            dict: A dictionary mapping agent names to their assigned absolute goal positions.
                The goals are assigned to minimize the sum of distances between agents and goals.

        Note:
            The observation structure for each agent contains:
            - Agent position at indices [2:4]
            - Relative goal positions starting at index 4
            Goals are converted from relative to absolute positions before assignment.
        """
        # Get absolute goals positions
        goal_poses = None
        agent_poses = []
        for agent_name in observation:
            feedback = observation[agent_name]
            agent_pose = feedback[2:4]
            if goal_poses is None:
                limit = 4 + self.n_subagents * 2
                goal_poses = feedback[4:limit].reshape(-1, 2) + agent_pose
            agent_poses.append(agent_pose)
        agent_poses = np.array(agent_poses)

        # Assign goals
        distance_matrix = np.linalg.norm(
            agent_poses[:, np.newaxis] - goal_poses, axis=2
        )
        _, goals_idx = linear_sum_assignment(distance_matrix)
        goals = {}
        for goal_idx, agent_name in zip(goals_idx, observation):
            goals[agent_name] = goal_poses[goal_idx]  # type: ignore
        return goals


class SubAgent:
    """A class representing a sub-agent that can navigate towards a goal position.

    The SubAgent implements simple goal-directed navigation behavior using a heuristic
    approach based on comparing x and y distances to the goal.

    Attributes:
        goal (numpy.ndarray): Target position the agent should move towards.
        tolerance (float): Minimum distance threshold before considering a direction change.

    Methods:
        give_goal(goal: numpy.ndarray): Sets the target position for the agent.
        act(observation: numpy.ndarray) -> int: Determines movement action based on current position
            and goal.

    Returns:
        int: Action index where:
            0 = no movement
            1 = move left
            2 = move right
            3 = move down
            4 = move up
    """

    def __init__(self):
        self.goal = None
        self.tolerance = 0.0

    def give_goal(self, goal: np.ndarray):
        """
        Sets the goal state for the multiagent oracle.

        Args:
            goal (np.ndarray): The target state/position to be reached by the agents.

        Returns:
            None

        Note:
            The goal state is stored as an instance variable and will be used for subsequent
            calculations or decision-making processes.
        """
        self.goal = goal

    def act(self, observation: np.ndarray) -> int:
        """Move the agent towards a predefined goal using a heuristic approach.

        The function determines the next action by comparing the relative position
        between the agent's current position and its goal, prioritizing movement
        along the axis with the larger difference.

        Args:
            observation (np.ndarray): The current observation of the environment,
                where indices [2:4] contain the agent's current position.

        Returns:
            int: The action to take:
                0 - no movement
                1 - move left
                2 - move right
                3 - move down
                4 - move up

        Note:
            The function uses a tolerance parameter to prevent oscillations
            when the agent is close to the goal.
        """
        pose = observation[2:4]
        direction_to_goal = self.goal - pose

        # Select action based on direction (this is a heuristic approach)
        if abs(direction_to_goal[0]) > abs(direction_to_goal[1]) + self.tolerance:
            if direction_to_goal[0] > 0 + self.tolerance:
                return 2  # move_right
            elif direction_to_goal[0] < 0 - self.tolerance:
                return 1  # move_left
        elif abs(direction_to_goal[0]) < abs(direction_to_goal[1]):
            if direction_to_goal[1] > 0 + self.tolerance:
                return 4  # move_up
            elif direction_to_goal[1] < 0 - self.tolerance:
                return 3  # move_down
        return 0


class Agent(BaseAgent):
    """A hierarchical agent class that implements an Hand-designed heuristic for the MPE spread environemnt.

    This agent class implements a two-level hierarchical control structure where:
    1. A high-level goal assigner allocates goals to individual agents
    2. Low-level agents execute actions to reach their assigned goals

    The agent supports training in parallel environments and handles episode tracking,
    logging, and visualization through tensorboard.

    Attributes:
        args (Args): Configuration arguments for the agent
        device (torch.device): Device to run computations on (CPU/GPU)
        env (ParallelEnv): The multi-agent environment
        num_env_agents (int): Number of agents in the environment
        LL_agents (List[SubAgent]): List of low-level agents
        goal_assigner (GoalAssigner): The high-level goal assignment module
        assigned_goals (Optional[Dict]): Current goal assignments for agents
        trained (bool): Whether the agent has completed training

    Methods:
        seed(): Sets random seeds for reproducibility
        reset(): Resets the agent's internal state
        train(): Trains the agent in the environment
        act(): Generates actions for all agents based on observations
        load_agent(): Loads a saved agent state
        save_agent(): Saves the current agent state
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
        self.seed()

        # env setup
        self.env = env
        self.num_env_agents = len(self.env.possible_agents)

        self.LL_agents: List[SubAgent] = [
            SubAgent() for _ in range(self.num_env_agents)
        ]
        self.goal_assigner = GoalAssigner(self.num_env_agents)
        self.assigned_goals = None

    def seed(self):
        """Sets random seeds for reproducibility across different random number generators.

        This method initializes the random seeds for the Python random module, NumPy,
        and PyTorch to ensure reproducible results across runs. It also configures PyTorch's
        CUDNN backend determinism if specified in arguments.

        Args:
            None

        Returns:
            None

        Note:
            The method uses the seed value stored in self.args.seed for all random generators
            and sets torch.backends.cudnn.deterministic based on self.args.torch_deterministic
        """
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def reset(self):
        """
        Reset the MultiagentOracle environment.

        This method resets the goal assignments for the agents in the environment. After
        calling this method, no agent will have an assigned goal until new assignments
        are made.

        Returns:
            None
        """
        self.assigned_goals = None

    def train(
        self, env: Any, log_path: Path | str | None = None, run_name: str | None = None
    ):
        """Runs the oracle on the provided environment.

        This method implements the training loop for the oracle, collecting experience,
        logging metrics, and saving traces of the interactions.
        In practice there is no learning.
        The oracle coordinates multiple agents to achieve their goals in the environment by following hand-designed heuristic.

        Basically each subagent is assigned a goal in the environment and it moves directly towards it.

        Args:
            env (Any): The environment to train on. Must implement gym-like interface.
            log_path (Path | str | None, optional): Directory path to save logs and traces.
                Defaults to "runs" if None.
            run_name (str | None, optional): Name for this training run. If None, generates
                name using experiment name, seed and timestamp.

        Returns:
            None: The method updates the oracle's internal state and saves logs/traces to disk.

        The training process:
        1. Sets up logging directories and saves hyperparameters
        2. Runs episodes collecting experience until total_timesteps is reached
        3. For each step:
           - Gets actions from the oracle
           - Executes environment steps
           - Tracks returns and done conditions
           - Logs metrics and saves traces
        4. Marks oracle as trained upon completion

        Logs include:
        - Per-agent episode returns
        - Total episode returns
        - Hyperparameters
        - Full interaction traces if save_all_trace=True
        """
        if run_name is None:
            run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        if log_path is None:
            log_path = Path("runs")
        else:
            log_path = Path(log_path)

        args_dict = self.args.to_dict()  # type: ignore
        save_path = log_path / run_name / "training"
        os.makedirs(save_path, exist_ok=True)

        with open(save_path / "params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

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
        obs, infos = self.env.reset()
        self.assigned_goals = None
        done = False
        episodic_return = np.zeros(self.num_env_agents)
        if self.args.save_all_trace:
            trace = Trace()
        else:
            trace = RewardTrace()
        episode = 0
        all_done = 0

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if all_done == self.num_env_agents:
                obs, infos = self.env.reset()
                self.assigned_goals = None  # Reset the agents goals after each reset
                all_done = 0
                episodic_return = np.zeros(self.num_env_agents)
                trace.empty()
                episode += 1
            # ---------------------------

            # Get actions
            # ---------------------------
            actions = self.act(observation=obs)
            # ---------------------------

            # Perform action
            # ---------------------------
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            # ---------------------------

            # Save data and log
            # ---------------------------
            trace.add(
                actions=actions,
                observations=obs,
                rewards=reward,
                terminations=terminated,
                truncations=truncated,
                infos=info,
                episode=episode,
            )

            all_done = 0
            for agent_idx, agent_name in enumerate(obs):
                if terminated[agent_name] or truncated[agent_name]:
                    done = True
                    all_done += 1
                else:
                    done = False
                episodic_return[agent_idx] += reward[agent_name]
                if done:
                    if self.args.verbose:
                        print(
                            f"global_step={global_step}, Agent: {agent_name} - Ep. return={episodic_return[agent_idx]}"
                        )
                    writer.add_scalar(
                        f"returns/{agent_name}",
                        episodic_return[agent_idx],
                        global_step,
                    )

            if all_done == self.num_env_agents:
                writer.add_scalar(
                    "returns/total",
                    np.sum(episodic_return),
                    global_step,
                )
                if self.args.verbose:
                    print(
                        f"global_step={global_step} - Total ep return: {np.sum(episodic_return)}"
                    )
                if hasattr(trace, "add_final_obs"):
                    trace.add_final_obs(next_obs)
                trace.save_trace(save_path=save_path, episode=episode)
            # ---------------------------

            obs = next_obs

        writer.close()
        self.trained = True

    def act(self, observation: Dict[str, np.ndarray]) -> dict:
        """
        Execute the action for each agent based on their observations and assigned goals.

        This method assigns goals to agents if they haven't been assigned yet, and then
        gets actions from each low-level agent based on their individual observations.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary mapping agent names to their
                observations as numpy arrays.

        Returns:
            dict: Dictionary mapping agent names to their chosen actions.
        """
        # Update goals for subagents
        if self.assigned_goals is None:
            self.assigned_goals = self.goal_assigner.assign_goals(observation)
        for agent, goal in zip(self.LL_agents, self.assigned_goals):
            agent.give_goal(goal=self.assigned_goals[goal])

        actions = {}
        for agent_name, ll_agent in zip(observation, self.LL_agents):
            actions[agent_name] = ll_agent.act(observation=observation[agent_name])
        return actions

    def load_agent(self, *args, **kwargs) -> bool:
        """
        Placeholder method for loading an agent into the environment.

        Returns:
            bool: Always returns True as a placeholder implementation.

        Note:
            This method is here just for compatibility with the library. There is nothing to load for this agent.
        """
        return True

    def save_agent(self, save_path: str | Path, name: None | str = None):
        """
        Save the agent configuration to a file.

        Args:
            save_path (str | Path): Path where to save the agent configuration.
            name (None | str, optional): Name to use for the saved file.
                If None, uses default naming. Defaults to None.

        Returns:
            None

        Notes:
            This method is here just for compatibility with the library. There is nothing to save for this agent.
        """
        pass
