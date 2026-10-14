"""Check what LevelEnv stores for a manager acting every k lower-level steps.

Builds: real env (2 agents, reward = step index, 6 steps) -> bottom level (2 agents) -> middle level (1 manager, k).
Agents are mocks that record every store() call. Run against the original and the patched level_env.py:

    python test_interval_fix.py path/to/TAG_Framework/src
"""
import importlib.util
import sys
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as GymDict


def load_level_env(repo_src):
    """repo_src: path to the <repo>/src directory whose level_env.py should be tested."""
    sys.path.insert(0, repo_src)
    from tame.hierarchy.level_env import LevelEnv
    return LevelEnv


class ScriptedEnv:
    """Two agents, 6 steps, reward of agent i at step t is 10*i + t, observation = [t]."""

    def __init__(self, n_steps=6):
        self.agents = ["a0", "a1"]
        self.n_steps = n_steps

    def reset(self):
        self.t = 0
        return {a: np.array([0.0]) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        self.t += 1
        obs = {a: np.array([float(self.t)]) for a in self.agents}
        rew = {a: 10.0 * i + (self.t - 1) for i, a in enumerate(self.agents)}
        done = self.t >= self.n_steps
        return obs, rew, {a: done for a in self.agents}, {a: False for a in self.agents}, {a: {} for a in self.agents}


class RecordingAgent:
    def __init__(self, name, children, comm_dim):
        self.name = name
        self.children = children
        self.communication_space = GymDict({name: Box(-np.inf, np.inf, shape=[comm_dim])})
        self.observation_space = GymDict({c: Box(-np.inf, np.inf, shape=[1]) for c in children})
        self.stored = []
        self.acts = 0

    def act_train(self, observation, global_step):
        self.acts += 1
        return {c: self.acts for c in self.children}  # action = decision index

    def act(self, observation):
        return {c: 0 for c in self.children}

    def store(self, state, action=None, reward=None, done=None):
        self.stored.append(dict(state={k: np.asarray(v).tolist() for k, v in state.items()},
                                action=None if action is None else dict(action), reward=reward, done=done))

    def update_step(self, global_step, writer):
        pass

    def comm(self, observation):
        if isinstance(observation, dict):
            observation = np.concatenate([o for o in observation.values()])
        return observation

    def seed(self, s):
        pass


def build(LevelEnv, k):
    env = ScriptedEnv()
    # Bottom-level agents carry the names of the real-environment agents, as in ppo3.py
    bottom_agents = {a: RecordingAgent(a, [a], comm_dim=1) for a in env.agents}
    bottom = LevelEnv(
        name="bottom", agents=bottom_agents, env=env,
        uplinks={"m0": ["a0", "a1"]}, downlinks={"a0": ["a0"], "a1": ["a1"]},
        action_ifreq=1, trace_type="reward", concat_obs=True,
        action_space=GymDict({"m0": GymDict({"a0": Discrete(5), "a1": Discrete(5)})}),
    )
    mid_agents = {"m0": RecordingAgent("m0", ["a0", "a1"], comm_dim=2)}
    middle = LevelEnv(
        name="middle", agents=mid_agents, env=bottom,
        uplinks=None, downlinks={"m0": ["a0", "a1"]},
        action_ifreq=k, trace_type="reward",
    )
    return env, bottom, middle, mid_agents["m0"]


def run(LevelEnv, k):
    env, bottom, middle, manager = build(LevelEnv, k)
    middle.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = middle.step(action=None, training=True)
        done = all(term[a] or trunc[a] for a in term)
    return manager


if __name__ == "__main__":
    LevelEnv = load_level_env(sys.argv[1])
    for k in (1, 2, 3, 4):
        m = run(LevelEnv, k)
        rows = [r for r in m.stored if r["action"] is not None]
        print(f"\n=== k={k}: manager decided {m.acts} times, stored {len(rows)} transitions (+{len(m.stored)-len(rows)} terminal rows)")
        for r in rows:
            print(f"   obs_t={r['state']['a0'][0]:.0f}  action={r['action']['a0']}  reward={r['reward']:.0f}  done={r['done']}")
