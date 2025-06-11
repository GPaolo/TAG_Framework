import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tame.data_handling.trace import RewardTrace

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def filter_unexpected_fields(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        cleaned_kwargs = {
            key: value for key, value in kwargs.items() if key in expected_fields
        }
        original_init(self, *args, **cleaned_kwargs)

    cls.__init__ = new_init
    return cls


def evaluate(
    agent: Any,
    env: Any,
    save_path: Path | str | None = None,
    eval_runs: int = 1,
) -> pd.DataFrame:
    """
    Evaluates an agent's performance in a given environment and saves the results.

    This function runs evaluation episodes, renders the environment, saves frames as images,
    and creates a video of the agent's performance. It also tracks and saves reward traces.

    Args:
        agent: The agent to be evaluated. Must implement an `act` method that takes observations
            and returns actions.
        env: The environment to evaluate the agent in. Must follow gym-like interface with
            reset() and step() methods.
        save_path: Directory path where evaluation results will be saved.
        eval_runs: Number of evaluation episodes to run. Defaults to 1.

    Note:
        The function creates the following directory structure for each evaluation run:
        save_path/
            evaluations/
                eval_0/
                    images/
                        frame_0.png
                        ...
                    video.mp4
                    trace.pkl

        The function requires matplotlib for rendering and assumes the environment has a
        render() method that returns image frames.
    """

    if save_path is not None:
        fig, ax = plt.subplots()
        save_path = Path(save_path)

    trace = RewardTrace()

    for eval_run in range(eval_runs):
        if save_path is not None:
            eval_save_path = save_path / "evaluations" / f"eval_{str(eval_run)}"
            images_path = eval_save_path / "images"

            if not images_path.exists():
                os.makedirs(images_path)

        # Run Episode
        # ---------------------------
        obs, infos = env.reset()
        trace.empty()

        done = False
        ts = 0

        while not done:
            with torch.no_grad():
                actions = agent.act(obs)
            new_obs, rew, term, trunc, infos = env.step(actions)
            trace.add(
                actions=actions,
                observations=obs,
                rewards=rew,
                terminations=term,
                truncations=trunc,
                infos=infos,
                episode=0,
            )

            obs = new_obs
            done = bool(
                np.all([(te or tr) for te, tr in zip(term.values(), trunc.values())])
            )

            if save_path is not None:
                frame = env.render()
                ax.clear()  # type: ignore
                ax.imshow(frame)  # type: ignore
                plt.pause(0.01)  # Pause to update the plot
                plt.savefig(images_path / f"frame_{ts}.png")  # type: ignore
            ts += 1

        rewards = pd.DataFrame(trace.rewards)
        print(f"Eval episode: {eval_run} - TS: {ts} - Total Reward:")
        print(rewards.sum())

        if hasattr(trace, "add_final_obs"):
            trace.add_final_obs(obs)

        if save_path is not None:
            trace.save_trace(save_path=eval_save_path)  # type: ignore

            make_video(input_path=eval_save_path / "images", output_path=eval_save_path)  # type: ignore

    return rewards  # type: ignore
    # ---------------------------


def make_video(input_path: str | Path, output_path: str | Path, prefix: str = "frame"):
    """
    Creates a video from a sequence of PNG frames using ffmpeg and removes the input directory.

    Args:
        input_path (str | Path): Directory containing the input PNG frames
        output_path (str | Path): Directory where the output video will be saved
        prefix (str, optional): Prefix of the frame filenames. Defaults to "frame"

    Notes:
        - Input frames should be named as "{prefix}_1.png", "{prefix}_2.png", etc.
        - Frames will be combined at 30 fps
        - Output video will be encoded using H.264 codec in MP4 format
        - The input directory will be deleted after video creation
        - Output video will be named "video.mp4"

    Example:
        >>> make_video("/path/to/frames", "/path/to/output", "frame")
        Video /path/to/output has been created successfully.
    """
    command = f"ffmpeg -framerate 30 -i {str(input_path)}/{prefix}_%d.png -c:v libx264 -pix_fmt yuv420p {str(output_path)}/video.mp4"
    os.system(command=command)
    shutil.rmtree(input_path)
    print(f"Video {output_path} has been created successfully.")


def hasmethod(entity: Any, func_name: str) -> bool:
    """Check if an entity has a callable method with the given name.

    Args:
        entity (Any): The object to check for the method.
        func_name (str): The name of the method to look for.

    Returns:
        bool: True if the entity has a callable method with the given name, False otherwise.

    Example:
        >>> class MyClass:
        ...     def my_method(self):
        ...         pass
        >>> obj = MyClass()
        >>> hasmethod(obj, 'my_method')
        True
        >>> hasmethod(obj, 'non_existent')
        False
    """
    return hasattr(entity, func_name) and callable(getattr(entity, func_name))
