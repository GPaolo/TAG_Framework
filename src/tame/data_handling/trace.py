import os
import numpy as np
import pandas as pd
from pathlib import Path


class Trace:
    """
    A class to handle and store trace data for episodes, including actions, observations, rewards, truncations, terminations, and additional info.

    Attributes:
        name (str): The name of the trace.
        is_empty (bool): Indicates if the trace is empty.
        len (int): The length of the trace.
        _data (dict): A dictionary to store trace data.

    Methods:
        __len__():
            Returns the length of the trace.

        add(episode: int, **kwargs):
            Adds data to the trace for a given episode.

        add_final_obs(final_obs: dict):
            Adds final observations to the trace.

        empty():
            Empties the trace data.

        save_trace(save_path: str | Path, episode: int | str | None = None):
            Saves the trace data to a CSV file.

        convert_to_dict_of_list(elements_list: list) -> dict:
            Converts a list of elements to a dictionary of lists.

        convert_to_df(element: dict, col_name_prefix: None | str = None) -> pd.DataFrame:
            Converts a dictionary of lists to a pandas DataFrame.
    """

    def __init__(self, name: str = "trace"):
        """The constructor for the Trace class.

        Args:
            name (str): The name of the trace. Defaults to "trace".
        """
        self._data = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "truncations": [],
            "terminations": [],
            "infos": [],
            "episode": [],
        }
        self.name = name
        self.is_empty = True
        self.len = 0

    def __len__(self):
        """Returns the length of the trace.

        Returns:
            int: The length of the trace.
        """
        return self.len

    def add(self, episode: int, **kwargs):
        """Adds data to the trace for a given episode.

        Args:
            episode (int): The episode number.
            **kwargs: Additional data to be added to the trace.
        """
        self.is_empty = False
        self.len += 1
        self._data["episode"].append(episode)
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)

    def add_final_obs(self, final_obs: dict):
        """Adds final observations to the trace.

        Args:
            final_obs (dict): The final observations for the episode.
        """
        self._data["observations"].append(final_obs)
        self.len += 1

    def empty(self):
        """Empties the trace data."""
        self._data = {key: [] for key in self._data}
        self.is_empty = True
        self.len = 0

    def save_trace(self, save_path: str | Path, episode: int | str | None = None):
        """Saves the trace data to a CSV file.

        Args:
            save_path (str | Path): The path to save the trace data.
            episode (int | str | None): The episode number to include in the file name. Defaults to None.
        """
        save_path = Path(save_path) / "traces"
        if not save_path.exists():
            os.makedirs(save_path, exist_ok=True)
        if episode is not None:
            save_name = f"{self.name}_ep_{episode}.csv"
        else:
            save_name = f"{self.name}.csv"

        dfs = []
        for key, values in self._data.items():
            if values:  # Only process non-empty lists
                if key == "episode":
                    df = self.convert_to_df({"episode": values}, col_name_prefix=None)
                else:
                    element = self.convert_to_dict_of_list(elements_list=values)
                    df = self.convert_to_df(element, col_name_prefix=key)
                dfs.append(df)

        df = pd.concat(dfs, axis=1)
        df.to_csv(save_path / save_name)
        reward_names = [col_name for col_name in df.columns if "reward" in col_name]

        mean_reward = df[["episode"] + reward_names].groupby("episode").mean()
        total_reward = df[["episode"] + reward_names].groupby("episode").sum()
        mean_reward.to_csv(save_path / f"mean_reward_{save_name}")
        total_reward.to_csv(save_path / f"total_reward_{save_name}")

    @staticmethod
    def convert_to_dict_of_list(elements_list: list) -> dict:
        """Converts a list of elements to a dictionary of lists.

        Args:
            elements_list (list): A list of elements.

        Returns:
            dict: A dictionary of lists.
        """
        if not elements_list:
            return {}

        element = {}
        for element_ts in elements_list:
            for key in element_ts:
                if key not in element:
                    element[key] = []
                element[key].append(element_ts[key])
        return element

    @staticmethod
    def convert_to_df(
        element: dict, col_name_prefix: None | str = None
    ) -> pd.DataFrame:
        """Converts a dictionary of lists to a pandas DataFrame.

        Args:
            element (dict): A dictionary of lists.
            col_name_prefix (None | str, optional): A prefix to add to column names. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame.
        """
        df = pd.DataFrame()

        for key, arrays in element.items():
            array_2d = np.vstack(arrays)
            if array_2d.shape[1] == 1:
                columns = [f"{col_name_prefix}_{key}"] if col_name_prefix else [key]
            else:
                columns = [
                    f"{col_name_prefix}_{key}_{i}" if col_name_prefix else f"{key}_{i}"
                    for i in range(array_2d.shape[1])
                ]

            temp_df = pd.DataFrame(array_2d, columns=columns)
            df = pd.concat([df, temp_df], axis=1)
        return df

    def __getattr__(self, name: str):
        """Dynamic property access for data stored in _data.

        Returns numpy array for valid keys in _data, raises AttributeError otherwise.
        """
        if name in self._data:
            return self._data[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class RewardTrace(Trace):
    """A specialized Trace that only records rewards and episodes."""

    def __init__(self, name="reward_trace"):
        """The constructor for the RewardTrace class.

        Args:
            name (str): The name of the trace. Defaults to "reward_trace".
        """
        super().__init__(name=name)
        self._data = {
            "rewards": [],
            "episode": [],
        }

    def add(  # type: ignore
        self,
        rewards: dict,
        episode: int,
        **kwargs,
    ):
        """Adds data to the trace for a given episode.

        Args:
            rewards (dict): The rewards for the episode.
            episode (int): The episode number.
        """
        super().add(episode=episode, rewards=rewards)

    def add_final_obs(self, *args, **kwargs):
        """Adds final observations to the trace.
        It does nothing. Here just for compatibility.
        """
        pass
