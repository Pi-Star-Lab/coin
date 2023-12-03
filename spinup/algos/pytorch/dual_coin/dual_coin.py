from copy import deepcopy
from collections import deque
import time

import numpy as np
import torch
from torch.optim import Adam
import gym

import spinup.algos.pytorch.coin.core as core
from spinup.algos.pytorch.coin.coin import ReplayBuffer
from spinup.utils.logx import EpochLogger


def dual_coin(
    env_fn,
    q_net=core.DQNQFunction,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=1000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    q_lr=1e-3,
    grad_steps=1,
    max_grad_norm=10,
    batch_size=32,
    update_interval=100,
    num_test_episodes=0,
    max_ep_len=1000,
    bonus=0,
    bonus_freq=10000,
    log_freq=10,
    logger_kwargs=dict(),
    save_freq=5000,
    eps_disp=0.05,
    eps_b=0.1,
    env_seed=-1,
):
    """
    Continual Optimistic Initialization (COIN) with b_dual for discrete action spaces.
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
        max_grad_norm (float): Maximum value for gradient clipping.
        bonus (float): Bonus value.
        bonus_freq (int): Number of env interactions that should elapse between
            bonus value increments.
        log_freq (int): How often (episodes) to log training stats.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        eps_disp (float): Threshold for dispersion.
        eps_b (float): Epsilon for b_dual.
        env_seed (int): Environment seed if the baseline policy works only on a
            environment seed.
    """

    # COIN specific hyperparams
    cum_bonus = 0.0  # cumulative bonus

    ep_ret_buffer = deque(maxlen=100)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape

    # Create q-network module and target network
    q_net = q_net(env.observation_space, env.action_space, **ac_kwargs)
    q_net_targ = deepcopy(q_net)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in q_net_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [q_net.q])
    logger.log(f"\nNumber of parameters: \t q: {var_counts[0]}\n")

    # Set up function for computing DQN loss
    def compute_loss_q(data):
        o, a, r, cr, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["coin_rew"],
            data["obs2"],
            data["done"],
        )

        # Current Q-value estimates
        cur_q = q_net.q(o)

        logger.store(QVals=cur_q.detach().numpy())

        # Bellman backup for Q function
        with torch.no_grad():
            next_q = q_net_targ.q(o2)
            # Follow greedy policy: use the one with the highest value
            next_q, _ = next_q.max(dim=1)
            next_q = next_q.squeeze(-1)
            # 1-step TD target
            targ_q_coin_a = cr + gamma * (1 - d) * next_q

            # Target Q-value estimates
            targ_q_coin = q_net_targ.q(o)
            # Q-values of other actions must remain unchanged
            targ_q_coin = targ_q_coin.index_put_(
                tuple(torch.column_stack((torch.arange(a.shape[0]), a)).long().t()),
                targ_q_coin_a,
            )

            backup = targ_q_coin

        # MSE loss against Bellman backup
        loss_q = ((cur_q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(Qcoin=cur_q.detach().numpy())

        return loss_q, loss_info

    # Set up optimizers for q-function
    q_optimizer = Adam(q_net.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(q_net)

    def update(data, grad_steps):
        for _ in range(grad_steps):
            # First run one gradient descent step for Q.
            loss_q, loss_info = compute_loss_q(data)

            # Update Q coin
            q_optimizer.zero_grad()
            loss_q.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(q_net.q.parameters(), max_grad_norm)
            q_optimizer.step()

        # Record things
        logger.store(LossQcoin=loss_q.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(q_net.parameters(), q_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o):
        if len(o.shape) > 1:
            # For grayscale image input we add an axis for the channel
            o = torch.as_tensor(o, dtype=torch.float32).unsqueeze_(1)
        else:
            o = torch.as_tensor(o, dtype=torch.float32)
        # Select the greedy action
        return q_net.act(o)

    def test_agent():
        for j in range(num_test_episodes):
            if env_seed >= 0:
                test_env.seed(seed=env_seed)
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(o)
                if isinstance(a, np.ndarray):
                    a = a.item()
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def is_new_coin_iteration(logger):
        # If returns are fairly constant for the last few episodes
        if "EpRet" in logger.epoch_dict.keys():
            ep_ret_mean, ep_ret_std = logger.get_stats("EpRet")
            if ep_ret_std / (abs(ep_ret_mean) + 0.00001) < eps_disp:
                return True
            else:
                return False
        return False

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    if env_seed >= 0:
        env.seed(seed=env_seed)
    o, ep_ret, ep_len = env.reset(), 0, 0
    n_episodes = 0

    for t in range(total_steps):
        a = get_action(o)
        if isinstance(a, np.ndarray):
            a = a.item()
        # Step the env
        o2, r, d, _ = env.step(a)

        # Penalized reward for exploration
        cr = r - bonus

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        # Add experience to replay buffer
        replay_buffer.store(o, a, r, cr, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            n_episodes += 1
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_ret_buffer.append(ep_ret)

            if env_seed >= 0:
                env.seed(seed=env_seed)
            o, ep_ret, ep_len = env.reset(), 0, 0

            # Update bonus
            if n_episodes % 50 == 0 and is_new_coin_iteration(logger):
                # Compute the bonus
                if "QVals" in logger.epoch_dict.keys():
                    _, _, min_q, max_q = logger.get_stats("QVals", True)
                    bonus = 0.1 * (max_q - min_q) * (1 - gamma) + eps_b
                    # Update coin rewards in buffer
                    replay_buffer.update_coin_rewards(bonus)
                    cum_bonus += bonus

        # Logging
        # if n_episodes % log_freq == 0:
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            # Test the performance of the deterministic version of the agent.
            test_agent()

            logger.log_tabular("Epochs", epoch)
            logger.log_tabular("Episodes", n_episodes)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            if num_test_episodes > 0:
                logger.log_tabular("TestEpRet", with_min_and_max=True)
                logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("QVals", with_min_and_max=True)
            logger.log_tabular("Bonus", bonus)
            logger.log_tabular("CumBonus", cum_bonus)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()

        if len(replay_buffer) >= batch_size and (t + 1) % update_interval == 0:
            for _ in range(update_interval):
                batch = replay_buffer.sample_batch()
                update(data=batch, grad_steps=grad_steps)

        if (t + 1) % save_freq == 0:
            # Save model
            logger.save_state({"env": env}, None)

            # Save all the desired logs into npy files for plotting
            logger.save_log("EpRet")
            logger.save_log("EpLen")
            logger.save_log("Bonus")

    # Save all the desired logs into npy files for plotting
    logger.save_state({"env": env}, None)
    logger.save_log("EpRet")
    logger.save_log("EpLen")
    logger.save_log("Bonus")


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

    dual_coin(
        lambda: gym.make(args.env),
        q_net=core.DQNQFunction,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
