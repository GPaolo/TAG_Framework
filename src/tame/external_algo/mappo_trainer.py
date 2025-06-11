import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
from tame.external_algo.mappo import MAPPO_MPE
from pettingzoo.mpe import simple_spread_v3


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            x = np.array([x[agent] for agent in x.keys()])
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.device = args.device
        self.action_dim = args.action_dim
        self.discrete_actions = args.discrete_actions
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        if self.discrete_actions:
            action_buffer_shape = [self.batch_size, self.episode_limit, self.N]
        else:
            action_buffer_shape = [
                self.batch_size,
                self.episode_limit,
                self.N,
                self.action_dim,
            ]

        self.buffer = {
            "obs_n": np.zeros(
                [self.batch_size, self.episode_limit, self.N, self.obs_dim]
            ),
            "s": np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
            "v_n": np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
            "a_n": np.zeros(action_buffer_shape),
            "a_logprob_n": np.zeros([self.batch_size, self.episode_limit, self.N]),
            "r_n": np.zeros([self.batch_size, self.episode_limit, self.N]),
            "done_n": np.ones([self.batch_size, self.episode_limit, self.N]),
        }
        self.episode_num = 0

    def store_transition(
        self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
    ):
        self.buffer["obs_n"][self.episode_num][episode_step] = obs_n
        self.buffer["s"][self.episode_num][episode_step] = s
        self.buffer["v_n"][self.episode_num][episode_step] = v_n
        self.buffer["a_n"][self.episode_num][episode_step] = a_n
        self.buffer["a_logprob_n"][self.episode_num][episode_step] = a_logprob_n
        self.buffer["r_n"][self.episode_num][episode_step] = r_n
        self.buffer["done_n"][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer["v_n"][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == "a_n":
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long).to(
                    self.device
                )
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32).to(
                    self.device
                )
        return batch


def make_env(episode_limit, render_mode="None"):
    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=episode_limit,
        local_ratio=0.5,
        render_mode=render_mode,
        continuous_actions=False,
    )
    env.reset(seed=42)
    return env


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number

        # Set random seed
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create env
        self.env = make_env(
            self.args.episode_limit, render_mode=args.render_mode
        )  # Discrete action space
        self.args.N = self.env.max_num_agents  # The number of agents
        self.args.obs_dim_n = [
            self.env.observation_spaces[agent].shape[0] for agent in self.env.agents
        ]  # obs dimensions of N agents
        self.args.action_dim_n = [
            self.env.action_spaces[agent].n for agent in self.env.agents
        ]  # actions dimensions of N agents

        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[
            0
        ]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[
            0
        ]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(
            self.args.obs_dim_n
        )  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir="runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}".format(
                self.env_name, self.number, self.seed
            )
        )

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(
                shape=self.args.N, gamma=self.args.gamma
            )

    def run(
        self,
    ):
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps

            _, episode_steps = self.run_episode_mpe()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print(
            "total_steps:{} \t evaluate_reward:{}".format(
                self.total_steps, evaluate_reward
            )
        )
        self.writer.add_scalar(
            "evaluate_step_rewards_{}".format(self.env_name),
            evaluate_reward,
            global_step=self.total_steps,
        )
        # Save the rewards and models
        # np.save(
        #     "./data_train/MAPPO_env_{}_number_{}_seed_{}.npy".format(
        #         self.env_name, self.number, self.seed
        #     ),
        #     np.array(self.evaluate_rewards),
        # )
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        observations, infos = self.env.reset()

        obs_n = np.array([observations[agent] for agent in observations.keys()])
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If you use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(
                obs_n, evaluate=evaluate
            )  # Get actions and the corresponding log probabilities of N agents
            s = (
                obs_n.flatten()
            )  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

            # need to transit 'a_n' into dict
            actions = {}
            for i, agent in enumerate(self.env.agents):
                actions[agent] = a_n[i]

            obs_next_n, r_n, done_n, _, _ = self.env.step(actions)

            done_n = np.array([done_n[agent] for agent in done_n.keys()])
            episode_reward += r_n["agent_0"]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(
                    episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
                )

            obs_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hyperparameters Setting for MAPPO in MPE environment"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(3e6),
        help=" Maximum number of training steps",
    )
    parser.add_argument(
        "--episode_limit",
        type=int,
        default=25,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=int(5000),
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument(
        "--evaluate_times", type=float, default=3, help="Evaluate times"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (the number of episodes)"
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=8,
        help="Minibatch size (the number of episodes)",
    )
    parser.add_argument(
        "--rnn_hidden_dim",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the rnn",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the mlp",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument(
        "--use_adv_norm",
        type=bool,
        default=True,
        help="Trick 1:advantage normalization",
    )
    parser.add_argument(
        "--use_reward_norm",
        type=bool,
        default=True,
        help="Trick 3:reward normalization",
    )
    parser.add_argument(
        "--use_reward_scaling",
        type=bool,
        default=False,
        help="Trick 4:reward scaling. Here, we do not use it.",
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy"
    )
    parser.add_argument(
        "--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay"
    )
    parser.add_argument(
        "--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip"
    )
    parser.add_argument(
        "--use_orthogonal_init",
        type=bool,
        default=True,
        help="Trick 8: orthogonal initialization",
    )
    parser.add_argument(
        "--set_adam_eps",
        type=float,
        default=True,
        help="Trick 9: set Adam epsilon=1e-5",
    )
    parser.add_argument(
        "--use_relu",
        type=float,
        default=False,
        help="Whether to use relu, if False, we will use tanh",
    )
    parser.add_argument(
        "--use_rnn", type=bool, default=False, help="Whether to use RNN"
    )
    parser.add_argument(
        "--add_agent_id",
        type=float,
        default=False,
        help="Whether to add agent_id. Here, we do not use it.",
    )
    parser.add_argument(
        "--use_value_clip", type=float, default=False, help="Whether to use value clip."
    )
    parser.add_argument(
        "--render_mode", type=str, default="None", help="File path to my result"
    )

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    runner.run()
