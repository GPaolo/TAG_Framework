from pathlib import Path
import importlib.util
import ast
import importlib
from dataclasses import dataclass, fields
from typing import Any, Type
from dataclasses_json import dataclass_json
from tame.hierarchy.base_agent import BaseAgent
from pettingzoo import ParallelEnv


@dataclass_json
@dataclass
class ArgsInterface:
    """A base interface class for handling nested argument structures.

    This class provides functionality to synchronize fields between a main argument class
    and its nested subarguments. It ensures that when a field is updated in the main class,
    the corresponding fields in the nested subargument classes are updated accordingly.

    Attributes:
        subargs (list[str] | None): A list of attribute names that represent nested argument
            structures. If None, no synchronization is performed.

    Methods:
        __post_init__(): Initializes the synchronization of fields between main and nested arguments.
        _sync_nested_args(nested_args: Any, main_fields: dict): Synchronizes overlapping fields
            between main arguments and nested arguments.
        __setattr__(name, value): Overrides attribute setting to maintain synchronization
            between main and nested arguments.

    Example:
        class MyArgs(ArgsInterface):
            field1: int = 1
            nested: NestedArgs
            subargs = ['nested']
    """

    subargs: list[str] | None = None
    seed: int | None = None
    cuda: int = 0

    def __post_init__(self):
        # Get all field names from the main Args class
        main_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.subargs is not None:
            for subarg_name in self.subargs:
                subarg = getattr(self, subarg_name)
                self._sync_nested_args(subarg, main_fields)

    def _sync_nested_args(self, nested_args: Any, main_fields: dict):
        """Syncs the args from the main args to the mappo and ppo args"""
        # Get all fields from the nested args class
        nested_fields = fields(nested_args)
        # Check for overlapping fields and update them
        for field in nested_fields:
            if field.name in main_fields:
                setattr(nested_args, field.name, main_fields[field.name])

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Only sync if it's a field (not during initialization)
        if name in [f.name for f in fields(self)]:
            main_fields = {name: value}
            if self.subargs is not None:
                for subarg_name in self.subargs:
                    subarg = getattr(self, subarg_name)
                    self._sync_nested_args(subarg, main_fields)


@dataclass
class ExperimentConfig:
    """Configuration class for experiment setup.

    This dataclass defines the configuration parameters for running an experiment, including
    the agent and environment specifications, and various experimental parameters.

    Attributes:
        Agent (BaseAgent): The agent class to be used in the experiment.
        Env (ParallelEnv): The environment class for the experiment.
        agent_args (ArgsInterface): Interface containing agent-specific arguments.
        MAX_TS (int): Maximum number of timesteps for the experiment.
        TOTAL_AGENTS (int): Total number of agents in the experiment.
        EVAL_RUNS (int): Number of evaluation runs to perform.
        SAVE_PATH (Path): Path where experiment results will be saved.
        RUN_NAME (str): Name identifier for the experiment run.
        TRAIN (bool): Flag indicating whether to run in training mode.
    """

    Agent: Type[BaseAgent]
    Env: Type[ParallelEnv]
    agent_args: ArgsInterface

    MAX_TS: int
    TOTAL_AGENTS: int
    EVAL_RUNS: int
    SAVE_PATH: Path
    RUN_NAME: str
    TRAIN: bool


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load configuration from a Python file.

    This function loads a configuration from a Python file and creates an ExperimentConfig object.
    The config file should define variables that match the fields in ExperimentConfig.

    Args:
        config_path (Union[str, Path]): Path to the configuration file

    Returns:
        ExperimentConfig: Configuration object populated with values from the file

    Raises:
        FileNotFoundError: If the config file cannot be loaded

    Example:
        >>> config = load_config("experiments/my_config.py")
    """
    config_path = Path(config_path)

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is not None:
        config = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(config)  # type: ignore
    else:
        raise FileNotFoundError(f"Could not load config file from: {config_path}")
    config_dict = {
        key: getattr(config, key)
        for key in ExperimentConfig.__annotations__.keys()
        if hasattr(config, key)
    }

    return ExperimentConfig(**config_dict)


def save_config(
    config: ExperimentConfig,
    save_path: str | Path,
    original_path: str | Path | None = None,
) -> None:
    """
    This function takes a configuration object and saves it to a Python file, preserving any imports
    and comments from an original config file if provided. The configuration values are written as
    Python variable assignments.

    Args:
        config (ExperimentConfig): The configuration object containing settings to save
        save_path (Union[str, Path]): Directory path where the config.py file will be saved
        original_path (Optional[Union[str, Path]]): Path to original config file to preserve its structure.
                                                   If None, no imports/comments are preserved.

    Returns:
        None

    Example:
        >>> config = ExperimentConfig(...)
        >>> save_config(config, "output/dir", "original/config.py")
        # Creates output/dir/config.py with preserved structure from original/config.py
    """
    save_path = Path(save_path) / "config.py"

    if original_path is not None:
        with open(original_path, "r") as f:
            original_content = f.read()

        # Parse the original file
        tree = ast.parse(original_content)

        # Extract imports and keep their original text
        import_lines = []
        for line in original_content.split("\n"):
            if line.startswith("from ") or line.startswith("import "):
                import_lines.append(line)
    else:
        import_lines = []
        original_content = ""

    # Create the content of the new config file
    lines = []

    # Add imports
    lines.extend(import_lines)
    if import_lines:
        lines.append("")  # Add blank line after imports

    for field in fields(config):
        if field.name not in ["Agent", "Env", "Args"]:
            value = getattr(config, field.name)
            if isinstance(value, str):
                new_line = f'{field.name} = "{value}"'
            elif isinstance(value, Path):
                new_line = f'{field.name} = Path("{value}")'
            else:
                new_line = f"{field.name} = {value}"
            lines.append(new_line)

    # Write the content to the file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n")
