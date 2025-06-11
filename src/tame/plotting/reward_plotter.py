from tame.plotting.utils import find_seeds, get_concat_traces
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import json
from scipy.signal import savgol_filter
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Plotter:
    """A class for plotting and analyzing reward data from experiments.

    This class provides functionality to load, process, and visualize reward data from
    training and testing experiments. It supports both training and test reward plotting,
    with options for smoothing, confidence intervals, and standard deviation visualization.

        exp_path (str | Path): Path to the experiment directory containing the data
        trace_type (str): Type of trace to analyze (e.g., 'individual', 'team')

    Attributes:
        exp_path (Path): Converted path to the experiment directory
        trace_type (str): Type of trace being analyzed

    Methods:
        load_test_rewards: Load reward data from test evaluations
        load_train_rewards: Load reward data from training sessions
        load_rewards: Load both training and test rewards for multiple experiments
        plot: Create visualization of the reward data
        export_as_pdf: Export the generated plots to PDF format

    Example:
        ```python
        plotter = Plotter("path/to/experiments", "individual")
        experiments = {
            "exp1": {"agents": 4, "recalc": False},
            "exp2": {"agents": 6, "recalc": False}
        plot = plotter.plot(experiments, seeds="all", plot_ci=True)
        ```
    """

    def __init__(self, exp_path: str | Path, trace_type: str) -> None:
        self.exp_path = Path(exp_path)
        self.trace_type = trace_type

    @staticmethod
    def load_test_rewards(
        exp_folders: List[Path], trace_type: str = "mean", verbose: int = 0
    ) -> pd.DataFrame | None:
        """
        Load test rewards from evaluation folders and compute statistics across seed runs.

        This method loads reward traces from evaluation folders, processes them, and computes mean
        and standard deviation across different random seeds.

        Args:
            exp_folders (List[Path]): List of paths to experiment folders for different seeds
            trace_type (str): Type of trace to load (e.g. 'mean', 'total')
            verbose (int, optional): Verbosity level for error reporting. Defaults to 0.

        Returns:
            pd.DataFrame | None: DataFrame containing mean and std of rewards across seeds,
                with columns ['mean', 'std']. Returns None if no data could be loaded.

        Notes:
            - Expects a folder structure with evaluations in 'eval_X' folders
            - Looks for CSV files named '{trace_type}_reward_reward_trace.csv'
            - Handles both direct placement and 'traces' subfolder for reward files
            - Averages rewards across multiple evaluations within each seed
            - Computes statistics across seeds
        """
        seed_runs = []
        for seed_path in exp_folders:
            try:
                evaluation = 0
                eval_path = seed_path / "evaluations"
                eval_rewards = []
                while (eval_path / f"eval_{evaluation}").exists():
                    if (eval_path / f"eval_{evaluation}" / "traces").exists():
                        reward_df_path = eval_path / f"eval_{evaluation}" / "traces"
                    else:
                        reward_df_path = eval_path / f"eval_{evaluation}"
                    reward_df_path = (
                        reward_df_path / f"{trace_type}_reward_reward_trace.csv"
                    )

                    df = pd.read_csv(reward_df_path, index_col="episode")
                    eval_rewards.append(df)

                    evaluation += 1
                if len(eval_rewards):
                    eval_rewards = pd.concat(eval_rewards, axis=1)

                    rewards_names = list(eval_rewards.columns)
                    env_rewards = []
                    for r_name in rewards_names:
                        if "reward_env_" in r_name:
                            env_rewards.append(r_name)

                    if not env_rewards:
                        env_rewards = rewards_names

                    # Mean reward across evaluations in the same seed
                    seed_runs.append(eval_rewards.mean(axis=1))
            except Exception as e:
                if verbose >= 2:
                    print(f"Cannot load test data for {seed_path}")
                    print(e)

        if len(seed_runs):
            mean_reward = pd.concat(seed_runs, axis=1).mean(axis=1)
            std_reward = pd.concat(seed_runs, axis=1).std(axis=1)
            rewards = pd.concat([mean_reward, std_reward], axis=1)
            rewards.rename(columns={0: "mean", 1: "std"}, inplace=True)
            return rewards
        else:
            return None

    @staticmethod
    def load_train_rewards(
        exp_folders: List[Path],
        recalc: bool,
        trace_type: str,
        level: str | None = None,
        verbose: int = 0,
    ) -> pd.DataFrame | None:
        """Load and process training rewards from experiment folders.

        This method loads training rewards from multiple experiment folders, processes them,
        and calculates statistics across seeds including mean, standard deviation and confidence intervals.

        Args:
            exp_folders (List[Path]): List of paths to experiment folders
            recalc (bool): Whether to recalculate rewards even if cached files exist
            trace_type (str): Type of trace to load ('train', 'test', etc.)
            level (str | None, optional): Name of the level from where to load rewards from. Defaults to None.
            verbose (int, optional): Verbosity level for printing progress. Defaults to 0.

        Returns:
            pd.DataFrame | None: DataFrame containing reward statistics (mean, std, confidence intervals)
                                across seeds, or None if no valid rewards found.
                                Columns are ['mean', 'std', 'cil', 'ciu'] where:
                                - mean: Mean reward across seeds
                                - std: Standard deviation across seeds
                                - cil: Lower 95% confidence interval
                                - ciu: Upper 95% confidence interval

        Raises:
            FileNotFoundError: If interface_level.json is not found when level='env'.
            interface_level.json contains the name of the level of the hierarchy that interfaces with the env.
            It should be save by the agent and is structured like:
            {
                "name": "bottom"
            }
        """
        seed_runs = []
        lenghts = {}
        for seed_path in exp_folders:
            # Make path where to look for
            rew_df_path = seed_path / "training"
            if level is None:
                if (seed_path / "training" / "traces").exists():
                    rew_df_path = seed_path / "training" / "traces"
                else:
                    rew_df_path = seed_path / "training"  # For old exps
            elif level == "env":  # The true rewards are from the interface level
                if (seed_path / "interface_level.json").exists():
                    interface_level_name = json.load(
                        open(seed_path / "interface_level.json", "r")
                    )["name"]
                else:
                    raise FileNotFoundError(f"No interface_level.json at {seed_path}")
                rew_df_path = seed_path / "training" / interface_level_name / "traces"
            else:
                rew_df_path = seed_path / "training" / level / "traces"

            rew_df_path = rew_df_path / f"{trace_type}_reward.csv"
            if rew_df_path.exists() and not recalc:
                df = pd.read_csv(rew_df_path, index_col="episode")
            else:
                try:
                    df = get_concat_traces(
                        data_path=rew_df_path.parent,
                        file_basename=f"{trace_type}_reward_trace_ep",
                    )
                except ValueError:
                    try:
                        df = get_concat_traces(
                            data_path=rew_df_path.parent,
                            file_basename=f"{trace_type}_reward_reward_trace_ep",
                        )
                    except ValueError:
                        continue

                df.to_csv(rew_df_path)

            try:
                params = json.load(open(seed_path / "params.json"))
            except Exception as error:
                params = None
                print(f"Error loading params {seed_path}")
                print(error)

            if params is not None:
                total_ts = params.get("total_timesteps", np.inf)
                ep_length = params.get("episode_length", np.inf)
                if len(df) * ep_length < total_ts:
                    lenghts[seed_path] = f"{len(df)*ep_length}/{total_ts}"
                # pass

            rewards_names = list(df.columns)
            env_rewards = []
            for r_name in rewards_names:
                if "reward_env_" in r_name:
                    env_rewards.append(r_name)

            if not env_rewards:
                env_rewards = rewards_names

            seed_runs.append(df[env_rewards].sum(axis=1))
            if verbose >= 0:
                print("#", end="")

        if verbose >= 2:
            if len(lenghts):
                print()
            for seed, ts_done in lenghts.items():
                print(f"Seed: {seed} - {ts_done}")

        n = len(seed_runs)
        if n:
            mean_reward = pd.concat(seed_runs, axis=1).mean(axis=1)
            std_reward = pd.concat(seed_runs, axis=1).std(axis=1)

            standard_error = std_reward / np.sqrt(n)
            t_value = stats.t.ppf(
                0.975, df=n - 1
            )  # 0.975 because it's two-tailed (0.95 + 0.025)
            # Calculate confidence intervals
            ci_lower = mean_reward - t_value * standard_error
            ci_upper = mean_reward + t_value * standard_error

            rewards = pd.concat([mean_reward, std_reward, ci_lower, ci_upper], axis=1)
            rewards.rename(
                columns={0: "mean", 1: "std", 2: "cil", 3: "ciu"}, inplace=True
            )
            return rewards
        else:
            return None

    def load_rewards(
        self,
        experiments: dict,
        seeds: str | list,
        verbose: int = 0,
        load_test: bool = False,
        scale_by_agents: bool = True,
    ) -> Tuple[dict, dict]:
        """Load reward data from experiments.

        This method loads training and optionally test rewards from multiple experiments with different seeds.

        Args:
            experiments (dict): Dictionary containing experiment configurations. Each experiment should have:
                - 'agents': Number of agents in experiment
                - 'recalc': Boolean flag for recalculation
                - 'level': (Optional) Level specification
                - 'name': (Optional) Display name for the experiment
            seeds (Union[str, list]): Seeds to load, can be either a string pattern or list of specific seeds
            verbose (int, optional): Verbosity level for output. Defaults to 0.
            load_test (bool, optional): Whether to load test rewards. Defaults to False.
            scale_by_agents (bool, optional): Whether to scale rewards by number of agents. Defaults to True.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - training_rewards (dict): Dictionary mapping experiment names to training reward data
                - test_rewards (dict): Dictionary mapping experiment names to test reward data, or None if load_test=False

        Examples:
            >>> plotter = RewardPlotter()
            >>> experiments = {
            ...     "exp1": {"agents": 2, "recalc": False},
            ...     "exp2": {"agents": 3, "recalc": True}
            ... }
            >>> train_rewards, test_rewards = plotter.load_rewards(experiments, seeds="*")
        """
        # Find experiment folders depending on the seeds
        experiment_folders = {}
        for exp in experiments:
            exp_name = f"{exp}__a{experiments[exp]['agents']}"
            experiment_folders[exp] = find_seeds(
                exp_path=self.exp_path, exp_name=exp_name, seeds=seeds
            )

        # Get experiment data
        training_rewards = {}
        test_rewards = {}
        for exp in experiment_folders:
            if verbose >= 0:
                print(f"Loading {exp} - {len(experiment_folders[exp])} seeds: ", end="")
            exp_data = self.load_train_rewards(
                experiment_folders[exp],
                recalc=experiments[exp]["recalc"],
                trace_type=self.trace_type,
                level=experiments[exp].get("level", None),
                verbose=verbose,
            )
            if verbose >= 0:
                print("")

            exp_name = experiments[exp].get("name", None)
            if exp_name is None:
                exp_name = exp

            if exp_data is not None:
                # Scale by the number of agents
                if scale_by_agents:
                    training_rewards[exp_name] = exp_data / experiments[exp]["agents"]
                else:
                    training_rewards[exp_name] = exp_data
            else:
                print(f"No train data for {exp}")

            if load_test:
                test_data = self.load_test_rewards(
                    exp_folders=experiment_folders[exp],
                    trace_type=self.trace_type,
                    verbose=verbose,
                )
                # Scale by the number of agents
                if test_data is not None:
                    if scale_by_agents:
                        test_rewards[exp_name] = test_data / experiments[exp]["agents"]  # type: ignore
                    else:
                        test_rewards[exp_name] = test_data / experiments[exp]["agents"]  # type: ignore
                else:
                    print(f"No test data for {exp}")
            else:
                test_rewards = None

        return training_rewards, test_rewards  # type: ignore

    def _plot_rewards_training(
        self,
        experiment_data: dict,
        oracle_data: dict,
        smoothing_window: int = 101,
        poly_order: int = 3,
        plot_std: bool = False,
        plot_ci: bool = False,
        max_x: None | int = None,
        figsize: tuple = (12, 6),
        figure: Tuple[Figure, Axes] | None = None,
        style="classic",
    ) -> Figure:
        """
        Plot training rewards using matplotlib.

        Args:
            experiment_data: Dictionary containing experiment data with 'mean', 'std', 'cil', 'ciu' columns
            oracle_data: Dictionary containing oracle data
            smoothing_window: Window size for Savitzky-Golay filter
            poly_order: Polynomial order for Savitzky-Golay filter
            plot_std: Whether to plot standard deviation bands
            plot_ci: Whether to plot confidence interval bands
            max_x: Maximum number of episodes to plot
            figsize: Figure size tuple (width, height)

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if plot_ci and plot_std:
            raise ValueError("Can plot only one between CI and STD")

        if smoothing_window % 2 == 0:
            print("Smoothing window size is even. Making it odd.")
            smoothing_window += 1

        # Find the maximum number of episodes
        max_episodes = 0
        episodes = None
        for exp in experiment_data:
            if len(experiment_data[exp].index) > max_episodes:
                episodes = experiment_data[exp].index
                max_episodes = len(episodes)

        plt.style.use(style)
        plt.rcParams.update(
            {
                "font.size": 14,  # Base font size * scale
                "axes.titlesize": 14 * 1.5,
                "axes.labelsize": 12 * 1.5,
                "xtick.labelsize": 10 * 1.5,
                "ytick.labelsize": 10 * 1.5,
                "legend.fontsize": 10 * 1.5,
                # "axes.spines.top": False,
                # "axes.spines.right": False,
            }
        )

        # Create figure
        if figure is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figure

        # Set up color cycle (similar to Bokeh's Category20)
        colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_data)))  # type: ignore
        # colors = plt.cm.viridis(np.linspace(0, 0.8, max(len(experiment_data), 3)))

        # Plot oracle reference lines
        for oracle_name in oracle_data:
            ax.axhline(
                y=oracle_data[oracle_name].mean()["mean"],
                color="red",
                linestyle=":",
                alpha=1,
            )

        # Plot each experiment
        for i, exp in enumerate(experiment_data):
            exp_data = experiment_data[exp]
            valid_episodes = exp_data.index
            if max_x is not None:
                valid_episodes = valid_episodes[:max_x]

            # Apply smoothing
            mean_smoothed = savgol_filter(
                exp_data["mean"], smoothing_window, poly_order
            )
            std_smoothed = savgol_filter(exp_data["std"], smoothing_window, poly_order)
            if plot_ci:
                cil_smoothed = savgol_filter(
                    exp_data["cil"], smoothing_window, poly_order
                )
                ciu_smoothed = savgol_filter(
                    exp_data["ciu"], smoothing_window, poly_order
                )

            # Plot mean line
            line = ax.plot(
                valid_episodes,
                mean_smoothed[: len(valid_episodes)],
                label=exp,
                color=colors[i],
                linewidth=1,
                alpha=1,
            )

            # Plot standard deviation bands
            if plot_std:
                ax.fill_between(
                    valid_episodes,
                    mean_smoothed[: len(valid_episodes)]
                    + std_smoothed[: len(valid_episodes)],
                    mean_smoothed[: len(valid_episodes)]
                    - std_smoothed[: len(valid_episodes)],
                    color=colors[i],
                    alpha=0.1,
                    linewidth=0,
                )

            # Plot confidence interval bands
            if plot_ci:
                ax.fill_between(
                    valid_episodes,
                    ciu_smoothed[: len(valid_episodes)],  # type: ignore
                    cil_smoothed[: len(valid_episodes)],  # type: ignore
                    color=colors[i],
                    alpha=0.1,
                    linewidth=0,
                )

        # Customize plot
        ax.set_title(f"Mean Reward by Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(title="Experiments", loc="lower right")
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(bottom=-14)  # Set lower limit to 0

        # Enable interactive data cursor
        def on_hover(event):
            if event.inaxes == ax:
                cont = ax.contains(event)
                if cont[0]:
                    for line in ax.get_lines():
                        if line in cont[1]["artist"]:
                            x, y = line.get_data()
                            ind = cont[1]["ind"][0]
                            x_arr = np.asarray(x)
                            y_arr = np.asarray(y)
                            ax.set_title(f"Episode: {x_arr[ind]:.0f}, Reward: {y_arr[ind]:.2f}")
                            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        plt.tight_layout()
        return fig

    def _plot_rewards_testing(
        self,
        experiment_data: dict,
        oracle_data: dict | None,
        figure: Tuple[Figure, Axes] | None = None,
        figsize: tuple = (4, 4.5),
    ) -> Figure:
        """
        Create a matplotlib visualization of rewards during testing phase.

        Args:
            experiment_data (dict): Dictionary containing experiment data where keys are experiment names
                and values are pandas DataFrames with 'mean' and 'std' columns.
            oracle_data (dict | None): Dictionary containing oracle data for reference comparison.
                If provided, adds a horizontal reference line to the plot.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object containing the plot with:
                - Error bars showing standard deviation
                - Points showing mean values
                - Whiskers at the ends of error bars (caps)
                - Optional horizontal reference line if oracle_data is provided
                - Customized axes with rotated experiment names
        """

        # Here we select a list of colors; extend or change as needed.
        colors_palette = colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_data)))  # type: ignore
        num_experiments = max(len(experiment_data), 3)
        colors = colors_palette[:num_experiments]

        plot_data = {
            "experiment": [],
            "mean": [],
            "std": [],
            "index": [],
            "color": [],
        }
        i = 0
        for exp, df in experiment_data.items():
            try:
                mean_val = df["mean"].iloc[0]
                std_val = df["std"].iloc[0]
                plot_data["mean"].append(mean_val)
                plot_data["std"].append(std_val)
                plot_data["experiment"].append(exp)
                plot_data["index"].append(i)
                plot_data["color"].append(colors[i])
            except Exception:
                pass
            i += 1

        # Create the figure and axis with approximate pixel dimensions converted to inches (DPI=100)
        if figure is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figure

        # Plot error bars including whiskers (capsize adds whiskers)
        ax.errorbar(
            plot_data["index"],
            plot_data["mean"],
            yerr=plot_data["std"],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=5,
            markersize=6,
            linestyle="none",
        )

        # Optionally, plot each point with its associated color (if desired)
        for idx, mean_val, col in zip(
            plot_data["index"], plot_data["mean"], plot_data["color"]
        ):
            ax.plot(idx, mean_val, "o", color=col, markersize=8)

        # Add horizontal reference line if oracle_data provided
        if oracle_data is not None:
            reference_value = next(iter(oracle_data.values()))["mean"].iloc[0]
            ax.axhline(
                y=reference_value,
                color="red",
                linestyle="--",
                linewidth=2,
            )

        # Customize axes
        ax.set_title("Mean reward Test")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Mean Value")
        ax.set_xticks(plot_data["index"])
        ax.set_xticklabels(plot_data["experiment"], rotation=45)

        plt.tight_layout()
        return fig

    def plot(
        self,
        experiments: dict,
        seeds: str | list,
        smoothing_window: int = 101,
        poly_order: int = 3,
        notebook: bool = True,  # Ignored for matplotlib
        plot_std: bool = False,
        plot_ci: bool = False,
        verbose: int = 0,
        plot_test: bool = False,
        scale_by_agents: bool = True,
        max_x: int | None = None,
        style: str = "classic",
    ):
        """
        Plot training (and testing) rewards for multiple experiments using matplotlib.

        The method creates plots comparing different experiments, with special handling for "oracle" experiments
        that are plotted separately. It can show both training curves and test results side by side.

        Args:
            experiments (dict): Dictionary containing experiment data to plot.
            seeds (str | list): Seeds to load data from, can be a string pattern or list of specific seeds.
            smoothing_window (int, optional): Window size for smoothing the curves. Defaults to 101.
            poly_order (int, optional): Polynomial order for smoothing. Defaults to 3.
            notebook (bool, optional): (Ignored) For compatibility.
            plot_std (bool, optional): Whether to plot standard deviation. Defaults to False.
            plot_ci (bool, optional): Whether to plot confidence intervals. Defaults to False.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            plot_test (bool, optional): Whether to include test results plot. Defaults to False.
            scale_by_agents (bool, optional): Whether to scale rewards by number of agents. Defaults to True.
            max_x (int | None, optional): Maximum x-axis value. Defaults to None.
            style (str, optional): Plot style to use. Defaults to "classic".

        Returns:
            matplotlib.figure.Figure: The generated matplotlib figure.
        """
        if plot_ci and plot_std:
            raise ValueError("Can plot only one between CI and STD")

        oracle_experiments = {}
        other_experiments = {}
        for exp_name in experiments:
            if "oracle" in exp_name:
                oracle_experiments[exp_name] = experiments[exp_name]
            else:
                other_experiments[exp_name] = experiments[exp_name]

        experiment_data, test_data = self.load_rewards(
            experiments=other_experiments,
            seeds=seeds,
            verbose=verbose,
            load_test=plot_test,
            scale_by_agents=scale_by_agents,
        )
        if len(oracle_experiments):
            if verbose >= 0:
                print("Loading oracle experiments")
            oracle_data, oracle_test_data = self.load_rewards(
                experiments=oracle_experiments,
                seeds=seeds,
                verbose=verbose,
                load_test=plot_test,
                scale_by_agents=scale_by_agents,
            )
        else:
            oracle_data = {}
            oracle_test_data = None

        # Create a matplotlib figure with either one or two subplots
        if plot_test and test_data is not None:
            fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 6))
            # Plot training rewards on the left axis
            self._plot_rewards_training(
                experiment_data=experiment_data,
                oracle_data=oracle_data,
                smoothing_window=smoothing_window,
                poly_order=poly_order,
                plot_std=plot_std,
                plot_ci=plot_ci,
                max_x=max_x,
                style=style,
                figure=(fig, ax_train),
            )
            # Plot test rewards on the right axis
            self._plot_rewards_testing(
                experiment_data=test_data,
                oracle_data=oracle_test_data,
                figure=(fig, ax_test),
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_rewards_training(
                experiment_data=experiment_data,
                oracle_data=oracle_data,
                smoothing_window=smoothing_window,
                poly_order=poly_order,
                plot_std=plot_std,
                plot_ci=plot_ci,
                max_x=max_x,
                style=style,
                figure=(fig, ax),
            )
        plt.tight_layout()
        return fig
