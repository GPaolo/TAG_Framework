from pathlib import Path
from typing import List
import pandas as pd


def find_seeds(exp_path: Path, exp_name: str, seeds: str | list) -> List[Path]:
    """Find experiment directories for specific seeds.

    This function searches for experiment directories that match a given experiment name
    and specified seeds within the experiment path.

    Args:
        exp_path (Path): Base directory path where experiment folders are located.
        exp_name (str): Name of the experiment to search for.
        seeds (Union[str, list]): Either a list of seed numbers or "all" to find all seeds.

    Returns:
        List[Path]: List of Path objects for each matching experiment seed directory.

    Example:
        >>> find_seeds(Path("/experiments"), "test_exp", [1, 2])
        [Path("/experiments/test_exp__s1"), Path("/experiments/test_exp__s2")]
        >>> find_seeds(Path("/experiments"), "test_exp", "all")
        [Path("/experiments/test_exp__s1"), Path("/experiments/test_exp__s2"), ...]
    """
    seed_folders = []
    if isinstance(seeds, list):
        for seed in seeds:
            seed_folders.append(exp_path / f"{exp_name}__s{seed}")
    elif seeds == "all":
        for exp in exp_path.iterdir():
            name = "_".join(exp.name.split("_")[:-1])  # Just drop the seed
            if exp.is_dir() and exp_name == name:
                seed_folders.append(exp)
    return seed_folders


def get_concat_traces(data_path: Path, file_basename: str) -> pd.DataFrame:
    """Concatenates multiple csv traces into a single pandas DataFrame.

    This function reads multiple csv files with sequential episode numbers and combines them
    into a single DataFrame. The files should follow the naming pattern:
    '{file_basename}_{episode_number}.csv'

    Args:
        data_path (Path): Path to the directory containing the csv files
        file_basename (str): Base name of the csv files without episode number and extension

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all traces, indexed by episode number

    Example:
        If data_path contains files 'trace_0.csv', 'trace_1.csv', etc.,
        get_concat_traces(data_path, 'trace') will combine all these files
        into a single DataFrame.
    """
    episode = 0
    filepath = data_path / f"{file_basename}_{episode}.csv"
    dfs = []
    while filepath.exists():
        dfs.append(pd.read_csv(filepath))
        episode += 1
        filepath = data_path / f"{file_basename}_{episode}.csv"
    return pd.concat(dfs).set_index("episode")
