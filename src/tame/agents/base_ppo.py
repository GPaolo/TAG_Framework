import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium
import random
from pathlib import Path
from typing import Any, Tuple, List
from tqdm import tqdm
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import os
from functools import cached_property
import time
from tame.utils.utils import filter_unexpected_fields
from tame.utils.config import ArgsInterface


@filter_unexpected_fields
@dataclass_json
@dataclass
class Args(ArgsInterface):
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int | None = 1
    torch_deterministic: bool = True
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
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    verbose: bool = False
    lean_comm: bool = False

    @cached_property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ContinuousActionAgent(nn.Module):
    def __init__(
        self,
        obs_size: int,
        actions_size: int,
        device: torch.device,
        torch_compile: bool = True,
    ):
        super().__init__()
        self.device = device
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, actions_size), std=0.01),
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, actions_size)).to(self.device)
        if torch_compile:
            self.critic = torch.compile(self.critic)
            self.actor_mean = torch.compile(self.actor_mean)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_probs, entropy = self.get_action(x, action=action)
        value = self.get_value(x)
        return action, log_probs, entropy, value


class DiscreteActionAgent(nn.Module):
    def __init__(
        self,
        obs_size: int,
        actions_size: int,
        device: torch.device,
        torch_compile: bool = True,
    ):
        super().__init__()
        self.device = device
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, actions_size), std=0.01),
        ).to(self.device)
        if torch_compile:
            self.critic = torch.compile(self.critic)
            self.actor = torch.compile(self.actor)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        # Squeeze needed otherwise broadcasting fucks up dimensions
        log_prob = probs.log_prob(action.squeeze())
        return action, log_prob, probs.entropy()

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_probs, entropy = self.get_action(x, action=action)
        value = self.get_value(x)
        return action, log_probs, entropy, value


class PPO:
    """A Proximal Policy Optimization (PPO) agent that can handle both discrete and continuous action spaces.
    Implementation coming from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

    It uses actor-critic networks under the hood and supports various features such as experience buffering,
    advantage computation via GAE (Generalized Advantage Estimation), learning rate annealing, and optional
    policy/value clipping in PPO updates.

    Arguments:
        observation_space (gymnasium.Space):
            The space representing the type, shape, and bounds of observations.
        action_space (gymnasium.Space):
            The space representing the type, shape, and bounds of actions.
        args (Args):
            A container of hyperparameters controlling aspects like learning rate, gamma, gae_lambda, etc.
        device (torch.device):
            The Torch device (CPU or GPU) on which tensors should be allocated and computations performed.
        name (str, optional):
            An identifier for the agent.
        torch_compile (bool, optional):
            If True, attempts to compile relevant networks for optimization.

    Attributes:
        observation_space (gymnasium.Space):
            Reference to the environment's observation space.
        action_space (gymnasium.Space):
            Reference to the environment’s action space.
        discrete (bool):
            Flag indicating if the action space is discrete or continuous.
        actor_critic (Union[ContinuousActionAgent, DiscreteActionAgent]):
            Neural network(s) used to produce actions and value estimates.
        optimizer (torch.optim.Optimizer):
            The optimizer for parameter updates (Adam in this case).
        buffer_size (int):
            Maximum capacity for the experience buffer.
        states (torch.Tensor):
            Buffer for storing observed states.
        actions (torch.Tensor):
            Buffer for storing performed actions.
        rewards (torch.Tensor):
            Buffer for storing immediate rewards received after actions.
        log_probs (torch.Tensor):
            Buffer for storing the log probabilities of actions.
        dones (torch.Tensor):
            Buffer for tracking whether an episode terminated at each step.
        buffer_idx (int):
            Current write index into the buffer.
        valid_mask (torch.Tensor):
            Mask to keep track of valid entries in the buffer (e.g. after the final state of an episode).
        update_counter (int):
            Number of updates performed so far.
        num_updates (int):
            Expected total number of training updates.

    Methods:
        reset_buffer():
            Clears the buffer indices and resets validity masks.
        store(state, action, reward, done):
            Records a transition in the replay buffer, respecting capacity limits.
        act(observation):
            Returns a greedy (exploitation) action at inference time without exploration noise.
        act_train(observation, global_step):
            Sample an action from the policy using stochastic exploration during training.
        compute_advantages(values, rewards):
            Calculates advantages using GAE and resets advantages when episodes end.
        update_step(global_step, writer):
            Checks if the buffer is full and initiates a PPO update if needed.
        _update_step(global_step, writer):
            Performs the PPO training loop on the buffered experience, computing and applying gradients.
        train(env, log_path, run_name):
            Primary training loop for the agent, including data collection and periodic PPO updates.
        seed(seed):
            Sets the random seeds for reproducibility across all relevant libraries.
        save_agent(save_path, name):
            Saves the learned model parameters to a file for later reuse.
        load_agent(load_path, name):
            Loads model parameters from a file if it exists and updates the actor-critic accordingly.
        update_lr(iteration):
            Gradually reduces the learning rate over training to improve convergence stability."""

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        args: Args,
        device: torch.device,
        name: str = "ppo",
        torch_compile: bool = True,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(self.observation_space, gymnasium.spaces.Box)

        self.args = args

        self.device = device
        self.name = name

        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.discrete = False
            self.actor_critic = ContinuousActionAgent(
                obs_size=self.observation_space.shape[0],
                actions_size=self.action_space.shape[0],
                device=device,
                torch_compile=torch_compile,
            ).to(self.device)
            self.action_shape = self.action_space.shape
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.discrete = True
            self.actor_critic = DiscreteActionAgent(
                obs_size=self.observation_space.shape[0],
                actions_size=int(self.action_space.n),
                device=device,
                torch_compile=torch_compile,
            ).to(self.device)
            self.action_shape = (int(self.action_space.n),)
        else:
            raise TypeError(
                f"Action space is of type: {type(self.action_space)}. Only discrete and box are supported"
            )

        self.optimizer = optim.Adam(  # type: ignore
            self.actor_critic.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # Initialize buffer
        self.buffer_size = self.args.batch_size

        self.states = torch.zeros(
            (self.buffer_size, self.observation_space.shape[0]), dtype=torch.float32
        )
        if self.discrete:
            self.actions = torch.zeros((self.buffer_size, 1), dtype=torch.float32)
        else:
            self.actions = torch.zeros(
                (self.buffer_size, self.action_space.shape[0]),  # type: ignore
                dtype=torch.float32,  # type: ignore
            )
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        # self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32).to(
            self.device
        )
        self.dones = torch.zeros(self.buffer_size, dtype=torch.bool)
        self.buffer_idx = 0
        self.valid_mask = torch.ones(self.buffer_size, dtype=torch.bool)

        self.reset_buffer()
        self.update_counter = 0
        # How many updates will be done
        self.num_updates = self.args.total_timesteps // self.args.batch_size

    def reset_buffer(self):
        """Reset the experience buffer"""
        # self.states: List[torch.Tensor] = []
        # self.actions: List[torch.Tensor | None] = []
        # self.rewards: List[float | None] = []
        # self.values: List[torch.Tensor] = []
        # self.log_probs: List[torch.Tensor | None] = []
        # self.dones: List[bool | None] = []
        # self.eop: List[int] = []  # Stores the idx of the element after done=True
        self.buffer_idx = 0
        self.valid_mask = torch.ones(self.buffer_size, dtype=torch.bool)

    def store(
        self,
        state: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor | None = None,
        reward: float | None = None,
        done: bool | None = None,
    ):
        """Store transition in buffer"""
        if self.buffer_idx >= self.buffer_size:
            # Do this rather than raise an error cause the level_env might try to add twice before training due to the last obs
            print(f"{self.name}: Buffer is full")
            return

        self.states[self.buffer_idx] = torch.FloatTensor(state)
        if action is not None:
            self.actions[self.buffer_idx] = torch.FloatTensor(action)
            self.log_probs[self.buffer_idx] = self.last_logprob
            self.valid_mask[self.buffer_idx] = True
        else:
            self.valid_mask[self.buffer_idx] = False

        if reward is not None:
            self.rewards[self.buffer_idx] = reward
        else:
            self.rewards[self.buffer_idx] = -1000
        if done is not None:
            self.dones[self.buffer_idx] = done

        self.buffer_idx += 1

    def act(self, observation: np.ndarray) -> int:
        obs = torch.Tensor(np.expand_dims(observation, axis=0)).to(self.device)
        with torch.no_grad():
            # The act functions for the actor_critics sample from the distribution.
            # But at inference we just want the max or the mean
            if self.discrete:
                action = (
                    torch.argmax(self.actor_critic.actor(obs), dim=1).cpu().numpy()[0]
                )
            else:
                action = self.actor_critic.actor_mean(obs).detach().cpu().numpy()
        return action

    def act_train(self, observation: np.ndarray, global_step: int) -> int:
        """Act with exploration during training"""
        obs = torch.Tensor(observation[np.newaxis, :]).to(self.device)
        with torch.no_grad():
            action, logprob, _ = self.actor_critic.get_action(obs)
            self.last_logprob = logprob
        return action.cpu().numpy()[0]

    def compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages using GAE
        I don't use done masking, but just skip the points after done=True
        """
        advantages = torch.zeros(self.buffer_size).to(self.device)
        gae = 0.0

        for t in reversed(range(self.buffer_idx)):
            # This is needed because of the end of episode storage
            if self.valid_mask[t]:
                if t < len(rewards) - 1 and not self.dones[t]:
                    next_value = values[t + 1].item()
                else:  # If done, we want to reset the gae as well (needed cause in the level env every stored sample is valid)
                    next_value = 0.0
                    gae = 0.0
                delta = rewards[t] + self.args.gamma * next_value - values[t].item()  # type: ignore
                gae = delta + self.args.gamma * self.args.gae_lambda * gae
                advantages[t] = gae
            else:
                gae = 0.0  # Reset for a new episode

        return advantages.unsqueeze(dim=-1)

    def update_step(self, global_step: int, writer: SummaryWriter | None):
        """The update is called only if the batch is full"""
        if self.buffer_idx >= self.buffer_size:
            if self.args.verbose:
                print(f"{self.name}: Updating models...")
            self._update_step(global_step=global_step, writer=writer)

    def _update_step(self, global_step: int, writer: SummaryWriter | None):
        """Perform PPO update"""
        self.update_counter += 1
        # Annealing the rate if instructed to do so.
        if self.args.anneal_lr:
            self.update_lr(iteration=self.update_counter)
            if writer is not None:
                writer.add_scalar(
                    f"{self.name}/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )

        # Compute advantages
        # ---------------------------
        states = self.states.to(self.device)
        with torch.no_grad():
            old_values = self.actor_critic.get_value(states)
        rewards = self.rewards.to(self.device)
        # [batch_size, 1]
        advantages = self.compute_advantages(old_values, rewards)

        # Prepare batch by dropping elements that are after the end of an episode
        # ---------------------------
        valid_mask = self.valid_mask.to(self.device)

        # NOTE we mask the advantages AFTER calculation cause before we need all the indexes
        # But we have to do it before normalization
        advantages = advantages[valid_mask]
        if self.args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # [batch_size, 1]
        actions = self.actions[self.valid_mask].to(self.device)
        # [batch_size, state_dim]
        states = states[valid_mask]
        # [batch_size,]
        old_log_probs = self.log_probs[valid_mask]
        # [batch_size, 1]
        old_values = old_values[valid_mask]

        # [batch_size, 1]
        returns = advantages + old_values
        # ---------------------------

        # PPO update for multiple epochs
        # ================================
        for update_ii in range(self.args.update_epochs):
            # Shorter than batch size cause of end-of-episodes
            data_length = int(self.valid_mask.sum())

            # Generate random mini-batches
            indices = np.random.permutation(data_length)

            # Minibatches
            # ================================
            for start_idx in range(0, data_length, self.args.minibatch_size):
                idx = indices[start_idx : start_idx + self.args.minibatch_size]

                # Get mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_values = old_values[idx]

                # Get current action probabilities and values
                (
                    _,
                    current_log_probs,
                    entropy,
                    current_values,
                ) = self.actor_critic.get_action_and_value(mb_states, action=mb_actions)

                # Compute ratio
                # [minibatch_size,]
                current_log_probs = torch.reshape(
                    current_log_probs, mb_old_log_probs.shape
                )
                # [minibatch_size,]
                logratio = current_log_probs - mb_old_log_probs
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Policy loss
                # [minibatch_size, 1]
                ratio = torch.reshape(ratio, mb_advantages.shape)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic loss
                if self.args.clip_vloss:
                    # [minibatch_size, 1]
                    critic_loss_unclipped = (current_values - mb_returns) ** 2
                    clipped_values = mb_values + torch.clamp(
                        current_values - mb_values,
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    critic_loss_clipped = (clipped_values - mb_returns) ** 2
                    critic_loss_max = torch.max(
                        critic_loss_unclipped, critic_loss_clipped
                    )
                    critic_loss = 0.5 * critic_loss_max.mean()
                else:
                    critic_loss = 0.5 * ((current_values - mb_returns) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = (
                    actor_loss
                    + self.args.vf_coef * critic_loss
                    - self.args.ent_coef * entropy_loss
                )

                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()

                # Log metrics if writer is provided
                if writer is not None:
                    # There will be multiple values written for the same global step. Maybe I can do an avg over the minibatches?
                    writer.add_scalar(
                        f"{self.name}/value_loss", critic_loss.item(), global_step
                    )
                    writer.add_scalar(
                        f"{self.name}/policy_loss", actor_loss.item(), global_step
                    )
                    writer.add_scalar(
                        f"{self.name}/entropy", entropy_loss.item(), global_step
                    )
                    writer.add_scalar(
                        f"{self.name}/old_approx_kl", old_approx_kl.item(), global_step
                    )
                    writer.add_scalar(
                        f"{self.name}/approx_kl", approx_kl.item(), global_step
                    )
            # ================================
        # ================================

        # Log explained variance
        y_pred, y_true = old_values.cpu().numpy(), returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if writer is not None:
            writer.add_scalar(
                f"{self.name}/explained_variance", explained_var, global_step
            )

        # Reset buffer after update
        self.reset_buffer()

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
        global_step = 0
        self.update_counter = 0
        obs, _ = env.reset(seed=self.args.seed)
        done = False
        episodic_return = 0

        for global_step in tqdm(
            range(self.args.total_timesteps), desc="Training step:"
        ):
            # Reset
            # ---------------------------
            if done:
                obs, _ = env.reset(seed=self.args.seed)
                done = False
                episodic_return = 0

            # Get actions
            # ---------------------------
            with torch.no_grad():
                obs = torch.Tensor([obs]).to(self.device)
                action, log_prob, _, value = self.actor_critic.get_action_and_value(obs)
            # ---------------------------

            # Act on env
            # ---------------------------
            next_obs, reward, terminations, truncations, infos = env.step(
                action.detach().cpu().numpy()
            )
            done = terminations or truncations
            # ---------------------------

            # Store transitions
            self.states.append(obs)  # type: ignore
            self.rewards.append(reward)  # type: ignore
            self.dones.append(done)  # type: ignore

            self.actions.append(action)  # type: ignore
            self.log_probs.append(log_prob)  # type: ignore
            self.values.append(value)  # type: ignore

            # Update state
            obs = next_obs
            episodic_return += reward

            if done:
                self.store(state=obs)
                print(
                    f"{self.name}: global_step={global_step}, - Ep. return={episodic_return}"
                )
                writer.add_scalar(f"returns/{self.name}", episodic_return, global_step)

            # Update models if batch is full
            self.update_step(global_step=global_step, writer=writer)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def save_agent(self, save_path: str | Path, name: None | str = None):
        save_path = Path(save_path)
        if name is None:
            name = "trained_model"
        model_save_path = save_path / "models" / f"{name}.pth"
        if not model_save_path.parent.exists():
            os.makedirs(model_save_path.parent)
        torch.save(self.actor_critic.state_dict(), model_save_path)
        if self.load_agent(save_path, name=name):
            print(f"{self.name}: model saved to {model_save_path}")
        else:
            print(f"{self.name}: Could not save the model!")

    def load_agent(self, load_path: Path | str, name: str = "trained_model") -> bool:
        load_path = Path(load_path) / "models" / f"{name}.pth"
        if load_path.exists():
            try:
                self.actor_critic.load_state_dict(torch.load(load_path))
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

    def update_lr(self, iteration: int):
        """Anneals the LR"""
        frac = 1.0 - (iteration - 1.0) / self.num_updates
        lrnow = frac * self.args.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow
