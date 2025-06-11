import numpy as np
from collections import deque
from dataclasses import dataclass
import torch
from typing import Tuple


@dataclass
class Batch:
    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        idx: np.ndarray,
    ):
        """Batch of transitions

        Args:
            observations (torch.Tensor): current observations
            actions (torch.Tensor): actions taken
            rewards (torch.Tensor): rewards received
            next_observations (torch.Tensor): next observations
            dones (torch.Tensor): whether episode is done
            idx (np.ndarray): indices of transitions
        """
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.dones = dones
        self.rewards = rewards
        self.idx = idx
        assert len(self.next_observations) == len(self.observations)
        assert len(self.actions) == len(self.observations)
        assert len(self.rewards) == len(self.observations)
        assert len(self.dones) == len(self.observations)
        assert len(self.idx) == len(self.idx)

    def __len__(self):
        """Return number of transitions in batch

        Returns:
            int: number of transitions
        """
        return len(self.observations)


class ReplayBuffer:
    """Replay buffer for storing transitions"""

    def __init__(
        self, capacity: int, seed: None | int = None, device: torch.device | None = None
    ):
        """Implements replay buffer for storing transitions

        Args:
            capacity (int): maximum number of transitions to store
            seed (None | int, optional): random seed. Defaults to None.
            device (torch.device | None, optional): torch device where to store transitions. Defaults to None.
        """
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.idx = deque(maxlen=capacity)

        self.capacity = capacity
        self.idx_lookup = {}
        self.rng = np.random.default_rng(seed=seed)
        self.end_of_ep = []
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

    def push(self, state, action, reward, done, idx):
        """Add transition to replay buffer

        Args:
            state (np.ndarray): current state
            action (np.ndarray): action taken
            reward (float): reward received
            done (bool): whether episode is done
            idx (int): index of transition
        """
        self.states.append(state)
        self.actions.append(np.atleast_1d(action) if action is not None else None)
        self.rewards.append(reward)
        self.dones.append(done)
        self.idx.append(idx)

    def sample(
        self, batch_size: int, with_replacement: bool = False, eop: bool = False
    ) -> Tuple[Batch, np.ndarray]:
        """Sample batch of transitions from replay buffer

        Args:
            batch_size (int): number of transitions to sample
            with_replacement (bool, optional): whether to sample with replacement. Defaults to False.
            eop (bool, optional): whether to sample transitions where done=True. Defaults to False.

        Returns:
            Tuple[Batch, np.ndarray]: batch of transitions and indices of sampled transitions
        """

        # Don't use the last added datapoint, as there is no next_state yet
        possible_idx = np.arange(len(self.states) - 1)

        # Remove next_state after done=True indexes
        self.eop_idx = np.array(
            [idx for idx, done in enumerate(self.dones) if done is None]
        )
        # If we don't want to use the last sars' we also remove the ts where done=True
        if not eop:
            self.eop_idx = list(self.eop_idx) + list(self.eop_idx - 1)

        possible_idx = list(set(possible_idx) - set(self.eop_idx))

        indices = self.rng.choice(
            possible_idx, size=batch_size, replace=with_replacement
        )
        data = self.retrieve_idx(indices=indices)
        return data, indices

    def retrieve_idx(self, indices: np.ndarray) -> Batch:
        """Retrieve transitions from replay buffer

        Args:
            indices (np.ndarray): indices of transitions to retrieve

        Returns:
            Batch: batch of transitions
        """
        states = []
        next_states = []
        actions = []
        dones = []
        rewards = []
        for idx in indices:
            states.append(torch.Tensor(self.states[idx]))
            next_states.append(torch.Tensor(self.states[idx + 1]))
            actions.append(torch.Tensor(self.actions[idx]))
            dones.append(torch.Tensor(np.atleast_1d(self.dones[idx])))
            rewards.append(torch.Tensor(np.atleast_1d(self.rewards[idx])))

        data = Batch(
            next_observations=torch.stack(next_states).to(self.device),
            observations=torch.stack(states).to(self.device),
            rewards=torch.stack(rewards).to(self.device),
            actions=torch.stack(actions).to(self.device),
            dones=torch.stack(dones).to(self.device),
            idx=indices,
        )
        return data

    def __len__(self):
        """Return number of transitions stored in replay buffer

        Returns:
            int: number of transitions
        """
        return len(self.states)
