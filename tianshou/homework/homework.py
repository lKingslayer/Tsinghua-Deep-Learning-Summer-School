#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @Contact  : huangsy1314@163.com
# @Website  : https://huangshiyu13.github.io
# @File    : homework

import os
import gym
import torch
import argparse
import numpy as np
from tianshou.env import VectorEnv
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from net import Net

def compute_return_base(batch, gamma=0.1):
    returns = np.zeros_like(batch.rew)
    last = 0
    for i in reversed(range(len(batch.rew))):
        returns[i] = batch.rew[i]
        if not batch.done[i]:
            returns[i] += last * gamma
        last = returns[i]
    batch.returns = returns
    return batch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def run_pg(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])

    test_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net = Net(
        args.layer_num, args.state_shape, args.action_shape,
        device=args.device, softmax=True)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PGPolicy(net, optim, dist, args.gamma,
                      reward_normalization=args.rew_norm)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    if not os.path.isdir(os.path.join(args.logdir)):
        os.mkdir(os.path.join(args.logdir))

    if not os.path.isdir(os.path.join(args.logdir, args.task)):
        os.mkdir(os.path.join(args.logdir, args.task))

    if not os.path.isdir(os.path.join(args.logdir, args.task, 'pg')):
        os.mkdir(os.path.join(args.logdir, args.task, 'pg'))

    log_path = os.path.join(args.logdir, args.task, 'pg')

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, save_fn=save_fn)

    train_collector.close()
    test_collector.close()

    if __name__ == '__main__':


        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()

        # Dimension of observation space and action space
        print("Dim of obervation space:", env.observation_space.shape[0])
        print("Dim of action space:", env.action_space)


if __name__ == '__main__':
    run_pg()
