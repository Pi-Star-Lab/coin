from copy import deepcopy
from collections import deque

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.coinr.core as core
import spinup.algos.pytorch.dqn.core as dqn_core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for COIN agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, 1), dtype=np.float32)
        self.act2_buf = np.zeros(core.combined_shape(size, 1), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.coin_rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.G_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, coin_rew, next_obs, next_act, done, G):
        """
        Add a batch of transitions to the buffer.
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act2_buf[self.ptr] = next_act
        self.rew_buf[self.ptr] = rew
        self.coin_rew_buf[self.ptr] = coin_rew
        self.done_buf[self.ptr] = done
        self.G_buf[self.ptr] = G
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        """
        Sample batch of transitions from the buffer.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            act2=self.act2_buf[idxs],
            rew=self.rew_buf[idxs],
            coin_rew=self.coin_rew_buf[idxs],
            done=self.done_buf[idxs],
            G=self.G_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def get_batch_by_indices(self, idxs):
        """
        Get batch of transitions at specified indices.
        """
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            act2=self.act2_buf[idxs],
            rew=self.rew_buf[idxs],
            coin_rew=self.coin_rew_buf[idxs],
            done=self.done_buf[idxs],
            G=self.G_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self):
        """
        Current size of the buffer.
        """
        return self.ptr

    def update_coin_rewards(self, bonus, gamma):
        """
        Update coin rewards.
        """
        for idx in range(self.ptr):
            self.coin_rew_buf[idx] -= bonus


def coinr(
    env_fn,
    q_net=core.COINQFunction,
    q_net_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    q_lr=1e-3,
    # batch_size=128,
    # update_interval=100,
    # num_test_episodes=0,
    # max_ep_len=1000,
    # grad_steps=1,
    # max_grad_norm=10,
    # bonus=1.0,
    # bonus_freq=10000,
    # training_starts=20,
    grad_steps=1,
    max_grad_norm=10,
    batch_size=32,
    update_interval=100,
    num_test_episodes=0,
    max_ep_len=1000,
    bonus=1,
    bonus_freq=10000,
    training_starts=20,
    base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_from_subopt_ll/dqn_from_subopt_ll_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/subopt_ll/subopt_ll_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_ll_run_1/dqn_ll_run_1_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/fourrooms_dqn_base/fourrooms_dqn_prior_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/coin_emptyrandom_b_1/coin_emptyrandom_b_1_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_multiroom_subopt/dqn_multiroom_subopt_s0/pyt_save/model.pt",
    # base_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_from_subopt_multiroom/dqn_from_subopt_multiroom_s0/pyt_save/model.pt",
    regret_bound=100,
    log_freq=10,
    logger_kwargs=dict(),
    save_freq=5000,
    env_seed=-1,
):
    """
    Continual Optimistic Initialization (COIN) for discrete action spaces.
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        q_net: The constructor method for a PyTorch Module with an ``act``
            method, and a ``q`` module. The ``act`` method should accept batches of
            observations as inputs, and ``q`` should accept a batch of observations
            as inputs. When called, these should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``q_coin``   (batch, act_dim)  | Tensor containing the current estimate
                                           | of Q* for the provided observations.
                                           | (Critical: make sure to
                                           | flatten this!)
            ``act``      (batch)           | Numpy array of actions for a batch of
                                           | observations derived from Q*.
            ===========  ================  ======================================
        q_net_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to DDPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)
        q_lr (float): Learning rate for Q-networks.
        batch_size (int): Minibatch size for SGD.
        update_interval (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        grad_steps (int): Number of gradient steps at each update.
        max_grad_norm (float): Maximum vlaue for gradient clipping.
        bonus (float): Bonus value.
        bonus_freq (int): Number of env interactions that should elapse between
            bonus value increment.
        training_starts (int): Number of episodes after which training starts.
        base_q_net_path (str): Path to the Q-net affiliated with the baseline
            suboptimal policy.
        regret_bound (float): Maximum allowed baseline regret per state and
            per exploratory action (w.r.t. the base suboptimal policy).
        log_freq (int): How often (episodes) to log training stats.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        env_seed (int): Environment seed if the baseline policy works only on a
            environment seed.
    """

    # COIN specific hyperparams
    cum_bonus = 0.0  # cumulative bonus
    cum_bonus += bonus

    # Performance of baseline policy
    base_perf = 0
    regret = 0  # actual total regret after adding bonus

    ep_ret_buffer = deque(maxlen=100)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape

    # Create q-network module and target network
    q_net = q_net(env.observation_space, env.action_space, **q_net_kwargs)

    # Get base q-network
    if base_q_net_path is not None:
        base_q_net = torch.load(base_q_net_path)
        if isinstance(base_q_net, core.COINQFunction):
            q_net.q_coin = deepcopy(base_q_net.q_coin)
        elif isinstance(base_q_net, dqn_core.DQNQFunction):
            q_net.q_coin = deepcopy(base_q_net.q)
    else:
        base_q_net = None

    q_net_targ = deepcopy(q_net)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in q_net_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [q_net.q_coin])
    logger.log(f"\nNumber of parameters: \t q: {var_counts[0]}\n")

    # Set up function for computing DQN loss
    def compute_loss_q(data):
        def weighted_mse_loss(pred, targ, weight):
            """
            MSE loss.
            """
            return torch.mean(weight * (pred - targ) ** 2)

        o, a, r, cr, o2, a2, d, G = (
            data["obs"],
            data["act"],
            data["rew"],
            data["coin_rew"],
            data["obs2"],
            data["act2"],
            data["done"],
            data["G"],
        )

        # Current Q-value estimates
        cur_q_coin = q_net.q_coin(o)
        cur_q_coin_a = torch.gather(cur_q_coin, dim=-1, index=a.long()).squeeze(-1)

        # Bellman backup for Q function
        with torch.no_grad():
            # Q-coin
            next_q_coin = q_net_targ.q_coin(o2)
            # Follow greedy policy: use the one with the highest value
            next_q_coin, _ = next_q_coin.max(dim=1)
            next_q_coin = next_q_coin.squeeze(-1)
            # 1-step TD target
            targ_q_coin_a = cr + gamma * (1 - d) * next_q_coin

            # Target Q-value estimates
            targ_q_coin = q_net_targ.q_coin(o)
            # Q-values of other actions must remain unchanged
            targ_q_coin = targ_q_coin.index_put_(
                tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                targ_q_coin_a,
            )
            backup_coin = targ_q_coin

            # Regret bound
            if base_q_net is not None:
                # If current action and base greedy action are same
                if isinstance(base_q_net, core.COINQFunction):
                    base_max_q, base_a = base_q_net.q_coin(o).max(dim=1)
                elif isinstance(base_q_net, dqn_core.DQNQFunction):
                    base_max_q, base_a = base_q_net.q(o).max(dim=1)
                is_base_act = (a.squeeze(-1) == base_a.float()).float().squeeze(-1)
                # Q-value of the base greedy action
                if isinstance(base_q_net, core.COINQFunction):
                    base_act_q = torch.gather(
                        base_q_net.q_coin(o), dim=-1, index=a.long()
                    ).squeeze(-1)
                elif isinstance(base_q_net, dqn_core.DQNQFunction):
                    base_act_q = torch.gather(
                        base_q_net.q(o), dim=-1, index=a.long()
                    ).squeeze(-1)
                # The regret gap to close
                # delta = base_max_q - bonus / (1 - gamma) - base_act_q
                # delta = base_max_q - (t / total_steps) * (bonus / (1 - gamma))
                delta = base_max_q - bonus / (1 - gamma)
                # eta = (base_max_q - G) / regret_bound
                eta = (base_perf - G) / regret_bound

                # Corrected eta
                # eta = (base_max_q - G) / (regret_bound - (base_max_q - G))
                # eta = (base_perf - G) / (regret_bound - (base_perf - G))

                # The Q-value we want to achieve
                targ_regret = delta

                # If the (avg.) return is greater than the base best action
                is_better_act = torch.gt(G, base_max_q).float().squeeze(-1)

                # If action is from the base policy, use the TD target
                # Elif action is not from base and has no regret, use the TD target
                # Else min of the regret target and TD target (implementation detail)
                backup_coin_a = (
                    is_base_act * targ_q_coin_a
                    + (1 - is_base_act) * is_better_act * targ_q_coin_a
                    + (1 - is_base_act)
                    * (1 - is_better_act)
                    * torch.minimum(targ_q_coin_a, targ_regret).squeeze(-1)
                )

                backup_coin = targ_q_coin.index_put_(
                    tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                    backup_coin_a,
                )

                mse_weights_a = (
                    is_base_act * torch.ones_like(targ_q_coin_a)
                    + (1 - is_base_act) * is_better_act * torch.ones_like(targ_q_coin_a)
                    + (1 - is_base_act)
                    * (1 - is_better_act)
                    * eta
                    * torch.ones_like(targ_q_coin_a)
                )

                mse_weights = torch.ones_like(targ_q_coin)
                mse_weights = mse_weights.index_put_(
                    tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                    mse_weights_a,
                )

        # MSE loss against modified Bellman backup
        if base_q_net is not None:
            loss_q_coin = weighted_mse_loss(cur_q_coin, backup_coin, mse_weights)
        else:
            loss_q_coin = ((cur_q_coin - backup_coin) ** 2).mean()

        # Useful info for logging
        loss_info = dict(Qcoin=cur_q_coin_a.detach().numpy())

        return loss_q_coin, loss_info

    # Set up optimizers for q-function
    q_coin_optimizer = Adam(q_net.q_coin.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(q_net)

    def update(data, grad_steps, start):
        for _ in range(grad_steps):
            # First run one gradient descent step for Q.
            loss_q_coin, loss_info = compute_loss_q(data)

            if start:
                # Update Q coin
                q_coin_optimizer.zero_grad()
                loss_q_coin.backward()
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(q_net.q_coin.parameters(), max_grad_norm)
                q_coin_optimizer.step()

        # Record things
        logger.store(LossQcoin=loss_q_coin.item(), **loss_info)

        if start:
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(q_net.parameters(), q_net_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o):
        # Select the greedy action
        return q_net.act(torch.as_tensor(o, dtype=torch.float32))

    def test_agent():
        for j in range(num_test_episodes):
            if env_seed >= 0:
                test_env.seed(seed=env_seed)
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def compute_regret(cur_return, base_perf):
        return max(base_perf - cur_return, 0)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    if env_seed >= 0:
        env.seed(seed=env_seed)
    o, ep_ret, ep_len = env.reset(), 0, 0
    n_episodes = 0
    rollout = []

    for t in range(total_steps):
        a = get_action(o)

        # Step the env
        o2, r, d, _ = env.step(a)

        # Penalized reward for exploration
        cr = r - bonus

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        a2 = get_action(o2)

        rollout.append((o, a, r, cr, o2, a2, d))

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # G computation
            G = [0] * len(rollout)
            G[-1] = rollout[-1][2]
            for i in range(len(rollout) - 2, -1, -1):
                G[i] = rollout[i][2] + gamma * G[i + 1]
            # Add all experiences in the episode to replay buffer
            for i, transition in enumerate(rollout):
                o, a, r, cr, o2, a2, d = transition
                replay_buffer.store(o, a, r, cr, o2, a2, d, G[i])

            rollout = []

            n_episodes += 1
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_ret_buffer.append(ep_ret)
            if n_episodes >= training_starts:
                regret += compute_regret(ep_ret, base_perf)
            else:
                # base_perf = 175  # for lunar lander
                if "ll" in base_q_net_path:
                    base_perf = 175
                else:
                    base_perf = np.mean(ep_ret_buffer)
            logger.store(Regret=regret)

            if env_seed >= 0:
                env.seed(seed=env_seed)

            o, ep_ret, ep_len = env.reset(), 0, 0

            # Logging
            if n_episodes % log_freq == 0:
                # Test the performance of the deterministic version of the agent.
                test_agent()

                logger.log_tabular("Episodes", n_episodes)
                logger.log_tabular("EpRet", with_min_and_max=True)
                logger.log_tabular("EpLen", average_only=True)
                if num_test_episodes > 0:
                    logger.log_tabular("TestEpRet", with_min_and_max=True)
                    logger.log_tabular("TestEpLen", average_only=True)
                logger.log_tabular("TotalEnvInteracts", t)
                logger.log_tabular("BasePerf", base_perf)
                logger.log_tabular("Bonus (b)", cum_bonus)
                logger.log_tabular("RegretBound (B)", regret_bound)
                logger.log_tabular("Regret", regret)
                logger.log_tabular("Time", time.time() - start_time)
                logger.dump_tabular()

        if len(replay_buffer) >= batch_size and (t + 1) % update_interval == 0:
            for _ in range(update_interval):
                batch = replay_buffer.sample_batch()
                update(
                    data=batch,
                    grad_steps=grad_steps,
                    start=n_episodes >= training_starts,
                )

        if (t + 1) % save_freq == 0:
            # Save model
            logger.save_state({"env": env}, None)

            # Save all the desired logs into npy files for plotting
            logger.save_log("EpRet")
            logger.save_log("EpLen")
            logger.save_log("Bonus")
            logger.save_log("Regret")

    # Save model
    logger.save_state({"env": env}, None)

    # Save all the desired logs into npy files for plotting
    logger.save_log("EpRet")
    logger.save_log("EpLen")
    logger.save_log("Bonus")
    logger.save_log("Regret")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="coin")
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    coinr(
        lambda: gym.make(args.env),
        q_net=core.COINQFunction,
        q_net_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
