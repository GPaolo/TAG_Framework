import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from dataclasses_json import dataclass_json
from tame.data_handling.replay_buffer import ReplayBuffer
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Any, Tuple
import gymnasium
from tame.utils.config import ArgsInterface


@dataclass_json
@dataclass
class Args(ArgsInterface):
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int | None = 1
    torch_deterministic: bool = True
    cuda: int = 1
    save_model: bool = True
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
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


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_size: int,
        actions_n: int,
        device: torch.device,
        torch_compile: bool = True,
    ):
        """This one has one output for each possible action combination"""
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, actions_n),
        ).to(self.device)
        if torch_compile:
            self.network = torch.compile(self.network)

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN:
    """Deep Q-Network (DQN) implementation for reinforcement learning.
    Implementation coming from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

    This class implements a DQN agent that can learn to make decisions in environments
    with discrete action spaces and continuous observation spaces. It includes features
    such as experience replay, target network, and epsilon-greedy exploration.

    Args:
        observation_space (gymnasium.Space): The observation space of the environment.
        action_space (gymnasium.Space): The action space of the environment.
        args (Args): Configuration parameters for the DQN agent.
        device (torch.device): The device (CPU/GPU) to run the computations on.
        name (str, optional): Name identifier for the DQN agent. Defaults to "dqn".
        torch_compile (bool, optional): Whether to compile the networks using torch.compile.
            Defaults to True.

    Attributes:
        observation_space (gymnasium.Space): The observation space of the environment.
        action_space (gymnasium.Space): The action space of the environment.
        obs_size (int): The size of the observation space.
        actions_n (int): The number of possible actions.
        q_net (QNetwork): The main Q-network for action value estimation.
        target_net (QNetwork): The target network for stable learning.
        rb (ReplayBuffer): Buffer for storing and sampling experiences.
        optimizer (torch.optim.Adam): The optimizer for training the Q-network.

    Methods:
        seed(seed): Sets random seeds for reproducibility.
        update_qnet(): Performs one gradient step on the Q-network.
        update_target_net(): Updates the target network parameters.
        update_step(global_step, writer): Performs a training step.
        train(env, log_path, run_name): Trains the DQN agent in the given environment.
        act_train(observation, global_step): Selects actions during training (with exploration).
        act(observation): Selects actions using the trained policy (without exploration).
        store(state, action, reward, done): Stores transitions in the replay buffer.
        save_agent(save_path, name): Saves the trained model to disk.
        load_agent(load_path, name): Loads a trained model from disk.

    Note:
        The implementation assumes a discrete action space and a continuous observation
        space. The environment must be compatible with the Gymnasium interface.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        args: Args,
        device: torch.device,
        name: str = "dqn",
        torch_compile: bool = True,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(self.observation_space, gymnasium.spaces.Box)
        assert isinstance(self.action_space, gymnasium.spaces.Discrete)

        self.obs_size = self.observation_space.shape[0]
        self.actions_n = int(self.action_space.n)
        self.args = args
        self.name = name
        self.device = device
        self.seed(seed=self.args.seed)

        self.q_net = QNetwork(
            obs_size=self.obs_size,
            actions_n=self.actions_n,
            device=self.device,
            torch_compile=torch_compile,
        )

        self.target_net = QNetwork(
            obs_size=self.obs_size,
            actions_n=self.actions_n,
            device=self.device,
            torch_compile=torch_compile,
        )

        self.rb = ReplayBuffer(
            capacity=self.args.buffer_size, seed=self.args.seed, device=self.device
        )
        self.rb_idx = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate)  # type: ignore
        self.target_net.load_state_dict(self.q_net.state_dict())

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def update_qnet(self) -> Tuple[np.ndarray, np.ndarray]:
        """This function performs one gradient step on the Qnet"""
        # Get Loss
        # ---------------------------
        data, _ = self.rb.sample(self.args.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_net(data.next_observations).max(dim=-1)
            td_target = data.rewards.flatten() + self.args.gamma * target_max * (
                1 - data.dones.flatten()
            )
        old_val = (
            self.q_net(data.observations)
            .gather(-1, data.actions.to(torch.int64))
            .squeeze()
        )
        loss = F.mse_loss(target=td_target, input=old_val)
        # ---------------------------

        # optimize the model
        # ---------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # -------------------------
        return loss.detach().cpu().numpy(), old_val.detach().cpu().numpy()

    def update_target_net(self):
        """This function updates the target network"""
        for target_net_p, q_net_p in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_net_p.data.copy_(
                self.args.tau * q_net_p.data + (1.0 - self.args.tau) * target_net_p.data
            )

    def update_step(self, global_step: int, writer: None | SummaryWriter):
        """This function performs a train step"""
        if global_step > self.args.learning_starts:
            if global_step % self.args.train_frequency == 0:
                loss, q_val = self.update_qnet()

                # Log
                # ---------------------------
                if global_step % 100 == 0 and writer is not None:
                    writer.add_scalar(f"{self.name}/td_loss", loss, global_step)
                    writer.add_scalar(
                        f"{self.name}/q_values", np.mean(q_val), global_step
                    )
                # --------------------------

            if global_step % self.args.target_network_frequency == 0:
                self.update_target_net()

    def train(
        self, env: Any, log_path: Path | str | None = None, run_name: str | None = None
    ):
        if run_name is None:
            run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        if log_path is None:
            log_path = Path("runs")
        else:
            log_path = Path(log_path)

        writer = SummaryWriter(log_path / run_name)
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
        obs, _ = env.reset(seed=self.args.seed)
        done = False
        episodic_return = 0
        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if done:
                obs, infos = env.reset(seed=self.args.seed)
                done = False
                episodic_return = 0
            # ---------------------------

            # Get actions
            # ---------------------------
            action = self.act_train(observation=obs, global_step=global_step)
            # ---------------------------

            # Perform action
            # ---------------------------
            next_obs, reward, terminated, truncated, info = env.step(action)
            # ---------------------------

            # Save data and log
            # ---------------------------
            episodic_return += reward
            done = terminated or truncated
            self.store(state=obs, action=action, reward=reward, done=done)

            if done:
                self.store(state=next_obs)  # store next obs
                print(f"global_step={global_step}, - Ep. return={episodic_return}")
                writer.add_scalar(f"returns/{self.name}", episodic_return, global_step)
            # ---------------------------

            obs = next_obs

            self.update_step(global_step=global_step, writer=writer)

        if self.args.save_model:
            save_path = log_path / run_name / "trained_policy.pth"
            torch.save(self.q_net.state_dict(), save_path)
            print(f"model saved to {save_path}")

        writer.close()
        self.trained = True

    def act_train(
        self,
        observation: np.ndarray,
        global_step: int,
    ) -> int:
        # Perform random exploration
        epsilon = linear_schedule(
            self.args.start_e,
            self.args.end_e,
            int(self.args.exploration_fraction * self.args.total_timesteps),
            global_step,
        )

        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            q_values = self.q_net(torch.Tensor([observation]).to(self.device))
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
        return action

    def act(self, observation: np.ndarray) -> int:
        q_values = self.q_net(torch.Tensor(np.array([observation])).to(self.device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
        return action

    def store(self, state, action=None, reward=None, done=None):
        self.rb.push(
            state=state, idx=self.rb_idx, action=action, reward=reward, done=done
        )
        self.rb_idx += 1

    def save_agent(self, save_path: str | Path, name: None | str = None):
        save_path = Path(save_path)
        if name is None:
            name = "trained_model"
        model_save_path = save_path / "models" / f"{name}.pth"
        if not model_save_path.parent.exists():
            os.makedirs(model_save_path.parent)
        torch.save(self.q_net.state_dict(), model_save_path)
        if self.load_agent(save_path, name=f"{name}.pth"):
            print(f"model saved to {model_save_path}")
        else:
            print("Could not save the model!")

    def load_agent(
        self, load_path: Path | str, name: str = "trained_model.pth"
    ) -> bool:
        load_path = Path(load_path) / "models" / name
        if load_path.exists():
            try:
                self.q_net.load_state_dict(torch.load(load_path))
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
