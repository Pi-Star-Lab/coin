from copy import deepcopy
from collections import deque
import time

import numpy as np
import torch
from torch.optim import Adam
import gym

import spinup.algos.pytorch.dqn.core as core
from spinup.algos.pytorch.dqn.dqn import ReplayBuffer
from spinup.utils.logx import EpochLogger


def reward_shift(
    env_fn,
    q_net=core.DQNQFunction,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
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
    bonus=0.5,
    epsilon_start=1.0,
    epsilon_end=0.07,
    epsilon_frac=0.2,
    training_starts=0,
    log_freq=10,
    logger_kwargs=dict(),
    save_freq=5000,
    env_seed=-1,
):
    """
    Reward Shifting (Optimistic Curiosity Exploration and Conservative Exploitation
    with Linear Reward Shaping, Sun et al. (Neurips-2022)) for discrete action space.
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
        bonus (float): Reward shift value.
        epsilon_start (float): Epsilon value at the start of training.
        epsilon_end (float): Final epsilon value.
        epsilon_frac (float): Fraction of total number of env interactions to
            reach the final epsilon value.
        training_starts (int): Episode number at which learning starts.
        grad_steps (int): Number of gradient steps at each update.
        max_grad_norm (float): Maximum value for gradient clipping.
        log_freq (int): How often (episodes) to log training stats.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        env_seed (int): Environment seed if the baseline policy works only on a
            environment seed.
    """

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

    def exploration_schedule(epsilon_start, frac_steps, epsilon_frac):
        return max(epsilon_end, epsilon_start * (1 - frac_steps / epsilon_frac))

    # Set up function for computing DQN loss
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        # Current Q-value estimates
        cur_q = q_net.q(o)
        # Extract Q-values for the actions in the buffer
        cur_q = torch.gather(cur_q, dim=1, index=a.long()).squeeze(-1)

        logger.store(QVals=cur_q.detach().numpy())

        # Bellman backup for Q function
        with torch.no_grad():
            next_q = q_net_targ.q(o2)
            # Follow greedy policy: use the one with the highest value
            next_q, _ = next_q.max(dim=1)
            next_q = next_q.squeeze(-1)
            # 1-step TD target
            backup = r - bonus + gamma * (1 - d) * next_q

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

    def get_action(o, deterministic=False):
        if len(o.shape) > 1:
            # For grayscale image input we add an axis for the channel
            o = torch.as_tensor(o, dtype=torch.float32).unsqueeze_(1)
        else:
            o = torch.as_tensor(o, dtype=torch.float32)
        if deterministic:
            return q_net.act(o)
        if np.random.rand() < exploration_rate:
            return np.random.choice(env.action_space.n)
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

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    if env_seed >= 0:
        env.seed(seed=env_seed)
    o, ep_ret, ep_len = env.reset(), 0, 0
    n_episodes = 0
    exploration_rate = epsilon_start

    for t in range(total_steps):
        a = get_action(o)
        if isinstance(a, np.ndarray):
            a = a.item()
        # Step the env
        o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        # Add experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Exploration schedule
        if n_episodes >= training_starts:
            exploration_rate = exploration_schedule(
                epsilon_start, t / total_steps, epsilon_frac
            )

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

            # Logging
            if n_episodes % log_freq == 0:
                # Test the performance of the deterministic version of the agent.
                test_agent()

                logger.log_tabular("Epochs", (t + 1) // steps_per_epoch)
                logger.log_tabular("Episodes", n_episodes)
                logger.log_tabular("Epsilon", exploration_rate)
                logger.log_tabular("EpRet", with_min_and_max=True)
                logger.log_tabular("EpLen", average_only=True)
                if num_test_episodes > 0:
                    logger.log_tabular("TestEpRet", with_min_and_max=True)
                    logger.log_tabular("TestEpLen", average_only=True)
                logger.log_tabular("TotalEnvInteracts", t)
                logger.log_tabular("QVals", with_min_and_max=True)
                logger.log_tabular("Bonus", bonus)
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

    # Save all the desired logs into npy files for plotting
    logger.save_state({"env": env}, None)
    logger.save_log("EpRet")
    logger.save_log("EpLen")


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

    reward_shift(
        lambda: gym.make(args.env),
        q_net=core.DQNQFunction,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
