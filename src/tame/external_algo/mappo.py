"""Multi-Agent Proximal Policy Optimization (MAPPO) Implementation

This module implements the Multi-Agent Proximal Policy Optimization algorithm for multi-agent reinforcement learning,
based on the implementation from: https://github.com/DaydayXtt/MAPPO-Pettingzoo-simple_spread_v3

The implementation includes:
- Actor-Critic architecture with both RNN (GRU) and MLP variants
- Several PPO optimization tricks like:
    - Orthogonal initialization
    - Advantage normalization
    - Value clipping
    - Gradient clipping
    - Learning rate decay
    - Entropy bonus

Key Components:
- Actor_RNN: Recurrent policy network using GRU
- Critic_RNN: Recurrent value network using GRU
- Actor_MLP: Feedforward policy network
- Critic_MLP: Feedforward value network
- MAPPO_MPE: Main MAPPO algorithm class for Multi-Particle Environments

The implementation supports:
- Discrete action spaces
- Partially observable multi-agent environments
- Optional agent IDs in observations
- Batch processing of episodes
- Model saving/loading
"""

import torch
import torch.nn as nn

# import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SequentialSampler


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN_D(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN_D, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        logits = self.fc2(self.rnn_hidden)
        distr = Categorical(logits=logits)
        action = distr.sample()
        return action, distr

    def get_logprob(self, action, distr):
        return distr.log_prob(action.squeeze())

    def get_entropy(self, distr, shape=None):
        if shape is None:
            return distr.entropy()
        else:
            new_logits = distr.logits.reshape(shape)
            new_distr = Categorical(logits=new_logits)
            return new_distr.entropy()


class Actor_RNN_C(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN_C, self).__init__()
        self.rnn_hidden = None
        self.actions_range = args.clip_cont_actions_range

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        self.actor_logstd = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        action_mean = self.fc2(self.rnn_hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        distr = Normal(action_mean, action_std)
        action = distr.sample()
        if self.actions_range is not None:
            action = torch.clamp(
                action, min=self.actions_range[0], max=self.actions_range[1]
            )
        return action, distr

    def get_logprob(self, action, distr):
        return distr.log_prob(action).sum(-1)

    def get_entropy(self, distr, shape=None):
        if shape is None:
            return distr.entropy().sum(-1)
        else:
            new_loc = distr.loc.reshape(shape)
            new_scale = distr.scale.reshape(shape)
            new_distr = Normal(new_loc, new_scale)
            return new_distr.entropy().sum(-1)


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP_D(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP_D, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        distr = Categorical(probs=prob)
        action = distr.sample()
        return action, distr

    def get_logprob(self, action, distr):
        return distr.log_prob(action.squeeze())

    def get_entropy(self, distr, shape=None):
        if shape is None:
            return distr.entropy()
        else:
            new_logits = distr.logits.reshape(shape)
            new_distr = Categorical(logits=new_logits)
            return new_distr.entropy()


class Actor_MLP_C(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP_C, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        self.actions_range = args.clip_cont_actions_range

        self.actor_logstd = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        action_mean = self.fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        distr = Normal(action_mean, action_std)
        action = distr.sample()
        if self.actions_range is not None:
            action = torch.clamp(
                action, min=self.actions_range[0], max=self.actions_range[1]
            )
        return action, distr

    def get_logprob(self, action, distr):
        return distr.log_prob(action).sum(-1)

    def get_entropy(self, distr, shape=None):
        if shape is None:
            return distr.entropy().sum(-1)
        else:
            new_loc = distr.loc.reshape(shape)
            new_scale = distr.scale.reshape(shape)
            new_distr = Normal(new_loc, new_scale)
            return new_distr.entropy().sum(-1)


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.discrete_actions = args.discrete_actions
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit

        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        self.clip_cont_actions_range = [-1, 1]
        self.name = args.name
        args.clip_cont_actions_range = self.clip_cont_actions_range

        self.device = args.device

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            if self.discrete_actions:
                self.actor = Actor_RNN_D(args, self.actor_input_dim).to(self.device)
            else:
                self.actor = Actor_RNN_C(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_RNN(args, self.critic_input_dim).to(self.device)
        else:
            if self.discrete_actions:
                self.actor = Actor_MLP_D(args, self.actor_input_dim).to(self.device)
            else:
                self.actor = Actor_MLP_C(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)

        if args.torch_compile:
            self.critic = torch.compile(self.critic)
            self.actor = torch.compile(self.actor)

        self.ac_parameters = list(self.actor.parameters()) + list(
            self.critic.parameters()
        )
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(  # type: ignore
                self.ac_parameters, lr=self.lr, eps=1e-5
            )
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)  # type: ignore

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32).to(
                self.device
            )  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N).to(self.device))

            # actor_input.shape=(N, actor_input_dim)
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(self.device)
            action, distr = self.actor(actor_inputs)  # prob is already a distribution
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                return action.detach().cpu().numpy(), None
            else:
                a_logprob_n = self.actor.get_logprob(action, distr)
                return action.detach().cpu().numpy(), a_logprob_n.detach().cpu().numpy()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = (
                torch.tensor(s, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(self.N, 1)
                .to(self.device)
            )  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N).to(self.device))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(
                self.device
            )  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.detach().cpu().numpy().flatten()

    def mask_tensor(self, tensor, mask):
        try:
            masked_tensor = tensor * mask
        except RuntimeError:
            masked_tensor = tensor * mask.unsqueeze(-1).expand_as(tensor)
        return masked_tensor

    def train(self, replay_buffer, total_steps, writer=None):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = (
                batch["r_n"]
                + self.gamma * batch["v_n"][:, 1:] * (1 - batch["done_n"])
                - batch["v_n"][:, :-1]
            )  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            # adv.shape(batch_size,episode_limit,N)
            adv = torch.stack(adv, dim=1)
            # v_target.shape(batch_size,episode_limit,N)
            v_target = adv + batch["v_n"][:, :-1]
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs, mask = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SequentialSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                # TODO add mask to RNN as well
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None  # type: ignore
                    self.critic.rnn_hidden = None  # type: ignore
                    entropy_target_shape = (self.mini_batch_size, self.N, -1)
                    a_logprob_n_now, values_now, dist_entropy = [], [], []
                    for t in range(self.episode_limit):
                        # dist.shape=(mini_batch_size*N, action_dim)
                        # Cause we treat each agent as a batch element
                        _, dist = self.actor(
                            actor_inputs[index, t].reshape(
                                self.mini_batch_size * self.N, -1
                            )
                        )
                        # action.shape = (mini_batch_size, N, action_dim)
                        action = batch["a_n"][index][:, t]
                        # logprob.shape = (mini_batch_size*N)
                        logprob = self.actor.get_logprob(
                            action=action.reshape(self.mini_batch_size * self.N, -1),
                            distr=dist,
                        )
                        a_logprob_n_now.append(
                            logprob.reshape(self.mini_batch_size, self.N)
                        )
                        dist_entropy.append(
                            self.actor.get_entropy(dist, shape=entropy_target_shape)
                        )  # dist_entropy.shape = (mini_batch_size, N)
                        v = self.critic(
                            critic_inputs[index, t].reshape(
                                self.mini_batch_size * self.N, -1
                            )
                        )  # v.shape=(mini_batch_size*N,1)
                        values_now.append(
                            v.reshape(self.mini_batch_size, self.N)
                        )  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    # dimension after stack (mini_batch_size, episode_limit, N)
                    a_logprob_n_now = torch.stack(a_logprob_n_now, dim=1).to(
                        self.device
                    )
                    dist_entropy = torch.stack(dist_entropy, dim=1).to(self.device)
                    values_now = torch.stack(values_now, dim=1).to(self.device)
                else:
                    _, dist_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                    # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                    dist_entropy = self.actor.get_entropy(distr=dist_now)
                    dist_entropy = self.mask_tensor(dist_entropy, mask[index])

                    # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                    # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                    try:
                        masked_actions = self.mask_tensor(
                            batch["a_n"][index], mask[index]
                        )
                        a_logprob_n_now = self.actor.get_logprob(
                            action=masked_actions, distr=dist_now
                        )
                        a_logprob_n_now = self.mask_tensor(a_logprob_n_now, mask[index])
                    except Exception as e:
                        print("AAAAAAAAAAAAAAAAAAAA")
                        raise e
                    # a_logprob_n_now = dist_now.log_prob(batch["a_n"][index])

                # a/b=exp(log(a)-log(b))
                # ratios.shape=(mini_batch_size, episode_limit, N)
                ratios = torch.exp(
                    a_logprob_n_now - batch["a_logprob_n"][index].detach()
                )
                ratios = self.mask_tensor(ratios, mask[index])

                surr1 = ratios * adv[index]
                surr1 = self.mask_tensor(surr1, mask[index])
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                surr2 = self.mask_tensor(surr2, mask[index])

                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = self.mask_tensor(actor_loss, mask[index])

                if self.use_value_clip:
                    values_old = self.mask_tensor(
                        batch["v_n"][index, :-1].detach(), mask[index]
                    )
                    values_error_clip = (
                        torch.clamp(
                            values_now - values_old, -self.epsilon, self.epsilon
                        )
                        + values_old
                        - v_target[index]
                    )
                    values_error_clip = self.mask_tensor(values_error_clip, mask[index])

                    values_error_original = values_now - v_target[index]
                    values_error_original = self.mask_tensor(
                        values_error_original, mask[index]
                    )

                    critic_loss = torch.max(
                        values_error_clip**2, values_error_original**2
                    )
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                critic_loss = self.mask_tensor(critic_loss, mask[index])

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if writer is not None:
            writer.add_scalar(
                f"{self.name}/actor_loss",
                actor_loss.mean(),
                total_steps,
            )

            writer.add_scalar(
                f"{self.name}/critic_loss",
                critic_loss.mean(),
                total_steps,
            )

            writer.add_scalar(
                f"{self.name}/total_loss",
                ac_loss,
                total_steps,
            )

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p["lr"] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch["obs_n"])
        critic_inputs.append(batch["s"].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = (
                torch.eye(self.N)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.batch_size, self.episode_limit, 1, 1)
            )
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(
            self.device
        )  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(
            self.device
        )  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)

        mask = ~batch["done_n"].clone().bool().detach()

        actor_inputs = actor_inputs * mask.unsqueeze(-1).expand_as(actor_inputs)
        critic_inputs = critic_inputs * mask.unsqueeze(-1).expand_as(critic_inputs)
        return actor_inputs, critic_inputs, mask

    def save_model(self, save_path, save_name):
        torch.save(self.actor.state_dict(), f"{save_path}/{save_name}.pth")

    def load_model(self, load_path, save_name):
        self.actor.load_state_dict(torch.load(f"{load_path}/{save_name}.pth"))
