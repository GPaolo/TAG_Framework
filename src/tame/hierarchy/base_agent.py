from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv
from torch.utils.tensorboard.writer import SummaryWriter

# Here we implement the two interfaces for agents
# BaseAgent is for generic agents that can be used standalone
# LevelAgent is for agents that can be used in a LevelEnv


class BaseAgent(ABC):
    """Base Agent class that defines the interface for all agents in the system.

    This abstract base class provides a standard interface that all agent implementations
    must follow. It defines core functionality like initialization, saving/loading agent
    state, taking actions based on observations, and training.

    Methods
    -------
    __init__(env: ParallelEnv, args: None | Any = None) -> None
        Initialize the agent with an environment and optional arguments.

    save_agent(save_path: str | Path, name: None | str = None)
        Save the agent's state to disk at the specified path.

    load_agent(load_path: Path | str, name: str = "trained_model") -> bool
        Load the agent's state from disk at the specified path.

    act(observation: Dict[str, np.ndarray]) -> dict
        Generate actions based on the current observation.

    train(env: Any, log_path: Path | str | None = None, run_name: str | None = None)
        Train the agent on the given environment.

    Attributes
    ----------
    None

    Notes
    -----
    All methods in this class are abstract and must be implemented by child classes.
    The agent is designed to work with parallel environments and uses dictionary-based
    observation/action spaces where keys are agent names.
    """

    @abstractmethod
    def __init__(self, env: ParallelEnv, args: None | Any = None) -> None:
        """Init the class"""

    @abstractmethod
    def save_agent(self, save_path: str | Path, name: None | str = None):
        """Saves the agent to disk.

        Args:
            save_path (str | Path): Path where to save the agent.
            name (str | None, optional): Name of the agent model file. Must not include '.pth' extension.
                If None, a default name will be used. Defaults to None.

        Notes:
            The agent models will be saved in the subfolder 'models' under the specified save_path,
            with the format: {save_path}/models/{name}.pth

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """

    @abstractmethod
    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        """
            Loads a trained agent model from disk.

        Args:
            load_path (Path | str): Path to the directory containing the agent's saved model
            name (str, optional): Name of the model file without the .pth extension. Defaults to "trained_model"

        Returns:
            bool: True if loading was successful, False otherwise

        Note:
            The actual model file will be loaded from {load_path}/models/{name}.pth.
            The name parameter should not include the .pth extension.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """

    @abstractmethod
    def act(self, observation: Dict[str, np.ndarray]) -> dict:
        """Act function

        Args:
            observation (Dict[str, np.ndarray]): {agent_name: observation}

        Returns:
            dict: {agent_name: action}
        """

    @abstractmethod
    def train(
        self, env: Any, log_path: Path | str | None = None, run_name: str | None = None
    ):
        """Trains the agent

        Args:
            log_path (Path | str | None, optional): Path to where log data will go. If None it will go to ./runs Defaults to None.
            run_name (str | None, optional): Name of the run. Defaults to None.
        """


class LevelAgent(BaseAgent):
    """Base class for agents that can be part of a hierarchical environment level.

    This abstract class defines the interface for agents that can be used within a hierarchical
    environment level. It provides the basic structure and requirements that all level agents
    must implement.

    Attributes:
        action_space (gymnasium.spaces.Dict): Dictionary of action spaces for the agent.
            Each key-value pair represents a different action component.
        observation_space (gymnasium.spaces.Dict): Dictionary of observation spaces for the agent.
            Each key-value pair represents a different observation component.
        communication_space (gymnasium.spaces.Dict | None): Dictionary of communication spaces for the agent.
            Defines the structure of communication with higher level agents. Can be None if no communication is needed.
        name (str): Identifier for the agent. Defaults to "base_agent".
        args (None | Any): Additional arguments for the agent. Defaults to None.
        torch_compile (bool): Whether to compile the agent's torch modules. Defaults to False.

    Notes:
        - Training is performed through update steps rather than a standalone train function
        - The class provides basic communication capabilities between hierarchical levels
        - All abstract methods must be implemented by concrete subclasses

    Example:
        ```python
        class CustomLevelAgent(LevelAgent):
            def __init__(self, observation_space, action_space, communication_space, device):
                super().__init__(observation_space, action_space, communication_space, device)
                # Additional initialization code
        ```
    """

    action_space: gymnasium.spaces.Dict  # {str: Box | Discrete}
    observation_space: gymnasium.spaces.Dict  # {str: Box | Discrete}
    communication_space: gymnasium.spaces.Dict | None  # {str: Box | Discrete}
    name: str = "base_agent"
    args: None | Any = None
    torch_compile: bool = False

    @abstractmethod
    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        action_space: gymnasium.spaces.Dict,
        communication_space: gymnasium.spaces.Dict | None,
        device,
        name: str = "base_agent",
        args: None | Any = None,
        torch_compile: bool = False,
    ) -> None:
        """Init class"""

    # def __init_subclass__(cls, **kwargs):
    #     """This method is called when a class inherits from LevelAgent"""
    #     super().__init_subclass__(**kwargs)

    #     original_init = cls.__init__

    #     @wraps(original_init)
    #     def new_init(self, *args, **kwargs):
    #         # Get the parameter names from the init signature
    #         sig = inspect.signature(original_init)
    #         bound_args = sig.bind(self, *args, **kwargs)
    #         bound_args.apply_defaults()

    #         # Check for env parameter
    #         if "env" in bound_args.arguments:
    #             raise ValueError(
    #                 f"Parameter 'env' is not allowed in {cls.__name__}'s __init__. "
    #                 "LevelAgent classes should not take an environment in their constructor."
    #             )

    #         # Call the original init if validation passes
    #         original_init(self, *args, **kwargs)

    #     cls.__init__ = new_init
    #     return cls

    @abstractmethod
    def update_step(self, global_step: int, writer: None | SummaryWriter):
        """Performs one update step of the agent. Called during level_env.step.

        This method is responsible for updating the agent's internal state and learning parameters.
        Some agents may perform multiple update steps when this method is called. Each agent is
        responsible for managing its own update frequency and logic.

        Args:
            global_step (int): Current global step count in the training process
            writer (None | SummaryWriter): TensorBoard SummaryWriter instance for logging metrics.
                If None, no metrics will be logged.

        Returns:
            None
        """

    @abstractmethod
    def store(self, state, action=None, reward=None, done=None):
        """
            This abstract method stores transition information in the agent's memory,
        which could be implemented as a replay buffer or other storage mechanism.

        Args:
            state: The state observation from the environment
            action (optional): The action taken by the agent
            reward (optional): The reward received from the environment
            done (optional): Boolean flag indicating if the episode terminated

        Returns:
            None
        """

    @abstractmethod
    def act_train(
        self,
        observation: Dict[str, np.ndarray],
        global_step: int,
    ) -> dict:
        """Act function called during train.
        Some agents perform exploration during training, hence the need for a different act function

        Args:
            observation (Dict[str, np.ndarray]): {agent_name: observation}
            global_step (int): Current global step

        Returns:
            dict: {agent_name: action}
        """

    @abstractmethod
    def seed(self, seed) -> None:
        """Seed the agent for reproducibility.

        Args:
            seed (int): The seed value to use for random number generators.

        Returns:
            None: This method doesn't return anything.

        Note:
            This is an abstract method that must be implemented by derived classes.
            It should set any random number generators used by the agent to the provided seed value.
        """

    def comm(self, observation: np.ndarray | Dict[str, np.ndarray]) -> np.ndarray:
        """Communication function. It generates a report of the agent state, to be sent to the higer level agent

        This function processes the agent's observation and prepares it for communication with a higher-level agent.
        If no communication space is defined for the agent, it raises a ValueError. For dictionary observations,
        it concatenates all values into a single numpy array.


        Args:
            observation (np.ndarray | Dict[str, np.ndarray]): Observation of the agent

        Returns:
            np.ndarray: Communication to be sent to the higher level agent

        Raises:
            ValueError: If the agent's communication space is not defined.
        """
        if self.communication_space is None:
            raise ValueError(
                f"Agent {self.name} has not communication space defined so it cannot generate comm."
            )
        if isinstance(observation, dict):
            observation = np.concat([obs for obs in observation.values()])
        return observation

    def train(
        self, env: Any, log_path: Path | str | None = None, run_name: str | None = None
    ):
        """
        Level agent does not need a train function, so we don't implement it.
        We add this here to implement the abstract method from the parent class
        The agent is trained through the train step.

        Args:
            env (Any): Environment to train the agent
            log_path (Path | str | None, optional): Path to where log data will go. If None it will go to ./runs Defaults to None.
            run_name (str | None, optional): Name of the run. Defaults to None.

        Raises:
            NotImplementedError: This method is not implemented for LevelAgent and needs to be implemented if used standalone.

        Notes:
            If you want to implement an agent that can be used both standalone and in a hierarchy, you should implement this function.
        """
        raise NotImplementedError(
            "LevelAgent does not implements a train function. Implement it if you want to use this agent standalone"
        )
