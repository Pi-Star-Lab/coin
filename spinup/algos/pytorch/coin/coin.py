from copy import deepcopy
from collections import deque

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.coin.core as core
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
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, coin_rew, next_obs, next_act, done):
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
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self):
        """
        Current size of the buffer.
        """
        return self.ptr

    def update_coin_rewards(self, bonus_val, gamma):
        """
        Update coin rewards.
        """
        for idx in range(self.ptr):
            self.coin_rew_buf[idx] -= bonus_val


def coin(
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
    num_test_episodes=0,
    batch_size=128,
    update_interval=100,
    max_ep_len=1000,
    bonus=1.0,
    bonus_freq=10000,
    regret_bound=100,
    log_freq=10,
    training_starts=20,
    prior_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_ll_run_1/dqn_ll_run_1_s0/pyt_save/model.pt",
    grad_steps=1,
    max_grad_norm=10,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Continual Optimistic Initialization (COIN) for discrete action spaces.
    """

    # COIN specific hyperparams
    cum_bonus = 0.0  # cumulative bonus

    # LL
    prior_ret = 0  # 175  # return of the policy to improve

    # Bridge-7
    # prior_ret = 9  # return of the policy to improve

    regret = 0  # actual total regret after adding bonus

    ep_ret_buffer = deque(maxlen=100)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape

    # Get prior q-network
    prior_q_net = torch.load(prior_q_net_path)

    # Create q-network module and target network
    q_net = q_net(env.observation_space, env.action_space, **q_net_kwargs)
    q_net.q_coin = deepcopy(prior_q_net.q)
    q_net.q_true = deepcopy(prior_q_net.q)
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
            return torch.sum(weight * (pred - targ) ** 2)

        o, a, r, cr, o2, a2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["coin_rew"],
            data["obs2"],
            data["act2"],
            data["done"],
        )

        # Current Q-value estimates
        cur_q_coin = q_net.q_coin(o)
        cur_q_coin_a = torch.gather(cur_q_coin, dim=-1, index=a.long()).squeeze(-1)

        # Current true Q-value estimates
        cur_q_true = q_net.q_true(o)
        cur_q_true_a = torch.gather(cur_q_true, dim=-1, index=a.long()).squeeze(-1)

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

            # Q-true
            next_q_true = q_net_targ.q_true(o2)
            # Retrieve the Q-values for the actions from the replay buffer
            next_q_true_a = torch.gather(next_q_true, dim=1, index=a2.long())
            next_q_true_a = next_q_true_a.squeeze(-1)
            # 1-step TD target
            targ_q_true_a = r + gamma * (1 - d) * next_q_true_a

            targ_q_true = q_net_targ.q_true(o)
            # Q-values of other actions must remain unchanged
            targ_q_true = targ_q_true.index_put_(
                tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                targ_q_true_a,
            )
            backup_true = targ_q_true

            # Regret bound
            if prior_q_net is not None:
                # If current action and prior greedy action are same
                prior_max_q, prior_a = prior_q_net.q(o).max(dim=1)
                is_prior_act = (a.squeeze(-1) == prior_a.float()).float().squeeze(-1)
                # Q-value of the prior greedy action
                prior_act_q = torch.gather(
                    prior_q_net.q(o), dim=-1, index=a.long()
                ).squeeze(-1)
                # The regret gap to close
                # delta = prior_max_q - bonus / (1 - gamma) - prior_act_q
                delta = prior_max_q - (t / total_steps) * (bonus / (1 - gamma))
                eta = (prior_max_q - cur_q_true_a) / regret_bound

                # The Q-value we want to achieve
                targ_regret = delta

                # If the (avg.) return is greater than the prior best action
                is_better_act = torch.gt(cur_q_true_a, prior_max_q).float().squeeze(-1)

                # If action is from the prior policy, use the TD target
                # Elif action is not from prior and has no regret, use the TD target
                # Else take the min of the TD target and regret target

                backup_coin_a = (
                    is_prior_act * targ_q_coin_a
                    + (1 - is_prior_act) * is_better_act * targ_q_coin_a
                    + (1 - is_prior_act)
                    * (1 - is_better_act)
                    * torch.minimum(targ_q_coin_a, targ_regret).squeeze(-1)
                )

                backup_coin = targ_q_coin.index_put_(
                    tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                    backup_coin_a,
                )

                mse_weights_a = (
                    is_prior_act * torch.ones_like(targ_q_coin_a)
                    + (1 - is_prior_act)
                    * is_better_act
                    * torch.ones_like(targ_q_coin_a)
                    + (1 - is_prior_act)
                    * (1 - is_better_act)
                    * eta
                    * torch.ones_like(targ_q_coin_a)
                )

                mse_weights = torch.ones_like(targ_q_coin)
                mse_weights = mse_weights.index_put_(
                    tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                    mse_weights_a,
                )

                # print(backup_coin)
                # print(mse_weights)

        # MSE loss against modified Bellman backup
        if prior_q_net is not None:
            loss_q_coin = weighted_mse_loss(cur_q_coin, backup_coin, mse_weights)
        else:
            loss_q_coin = ((cur_q_coin - backup_coin) ** 2).mean()

        # MSE loss
        loss_q_true = ((cur_q_true - backup_true) ** 2).mean()

        # Useful info for logging
        loss_info = dict(
            Qcoin=cur_q_coin_a.detach().numpy(), Qtrue=cur_q_true_a.detach().numpy()
        )

        return loss_q_coin, loss_q_true, loss_info

    # Set up optimizers for q-function
    q_coin_optimizer = Adam(q_net.q_coin.parameters(), lr=q_lr)
    q_true_optimizer = Adam(q_net.q_true.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(q_net)

    def update(data, grad_steps, start):
        for _ in range(grad_steps):
            # First run one gradient descent step for Q.
            loss_q_coin, loss_q_true, loss_info = compute_loss_q(data)

            if start:
                # Update Q coin
                q_coin_optimizer.zero_grad()
                loss_q_coin.backward()
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(q_net.q_coin.parameters(), max_grad_norm)
                q_coin_optimizer.step()

                # Update Q true
                q_true_optimizer.zero_grad()
                loss_q_true.backward()
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(q_net.q_true.parameters(), max_grad_norm)
                q_true_optimizer.step()

        # Record things
        logger.store(
            LossQcoin=loss_q_coin.item(), LossQtrue=loss_q_true.item(), **loss_info
        )

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
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def compute_regret(cur_return, prior_ret):
        return max(prior_ret - cur_return, 0)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    n_episodes = 0

    for t in range(total_steps):
        a = get_action(o)

        # Step the env
        o2, r, d, _ = env.step(a)

        # env.render()

        # Penalized reward for exploration
        cr = r - bonus

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        a2 = get_action(o2)

        # Add experience to replay buffer
        replay_buffer.store(o, a, r, cr, o2, a2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            n_episodes += 1
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_ret_buffer.append(ep_ret)
            if n_episodes >= training_starts:
                regret += compute_regret(ep_ret, prior_ret)
            else:
                prior_ret = np.mean(ep_ret_buffer)
            logger.store(Regret=regret)
            o, ep_ret, ep_len = env.reset(), 0, 0

            # Logging
            if n_episodes % log_freq == 0:
                logger.log_tabular("Episodes", n_episodes)
                logger.log_tabular("EpRet", with_min_and_max=True)
                # logger.log_tabular("TestEpRet", with_min_and_max=True)
                logger.log_tabular("EpLen", average_only=True)
                # logger.log_tabular("TestEpLen", average_only=True)
                logger.log_tabular("TotalEnvInteracts", t)
                logger.log_tabular("Qcoin", with_min_and_max=True)
                logger.log_tabular("Qtrue", with_min_and_max=True)
                logger.log_tabular("LossQcoin", average_only=True)
                logger.log_tabular("LossQtrue", average_only=True)
                logger.log_tabular("Bonus", bonus)
                logger.log_tabular("Regret", with_min_and_max=True)
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

        # # End of epoch handling
        # if (t + 1) % steps_per_epoch == 0:
        #     epoch = (t + 1) // steps_per_epoch

        #     # Test the performance of the deterministic version of the agent.
        #     test_agent()

        #     # Log info about epoch
        #     logger.log_tabular("Epoch", epoch)
        #     logger.log_tabular("EpRet", with_min_and_max=True)
        #     # logger.log_tabular("TestEpRet", with_min_and_max=True)
        #     logger.log_tabular("EpLen", average_only=True)
        #     # logger.log_tabular("TestEpLen", average_only=True)
        #     logger.log_tabular("TotalEnvInteracts", t)
        #     logger.log_tabular("Qcoin", with_min_and_max=True)
        #     logger.log_tabular("Qtrue", with_min_and_max=True)
        #     logger.log_tabular("LossQcoin", average_only=True)
        #     logger.log_tabular("LossQtrue", average_only=True)
        #     logger.log_tabular("Bonus", bonus)
        #     logger.log_tabular("Regret", with_min_and_max=True)
        #     logger.log_tabular("Time", time.time() - start_time)
        #     logger.dump_tabular()

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

    coin(
        lambda: gym.make(args.env),
        q_net=core.COINQFunction,
        q_net_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
