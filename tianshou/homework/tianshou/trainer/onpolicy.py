import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        repeat_per_collect: int,
        episode_per_test: Union[int, List[int]],
        batch_size: int,
        train_fn: Optional[Callable[[int], None]] = None,
        test_fn: Optional[Callable[[int], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        **kwargs
):

    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    test_in_train = train_collector.policy == policy

    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)

        if True:
            step_now = 0
            # while t.n < t.total:
            while step_now < step_per_epoch:
                print('Ep:{} {}/{}'.format(epoch,step_now,step_per_epoch))

                result = train_collector.collect(n_episode=collect_per_step,
                                                 log_fn=log_fn)
                data = {}

                if test_in_train and stop_fn and stop_fn(result['rew']):

                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)

                    if stop_fn and stop_fn(test_result['rew']):

                        if save_fn:

                            save_fn(policy)

                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        return
                    else:

                        policy.train()
                        if train_fn:
                            train_fn(epoch)
                losses = policy.learn(
                    train_collector.sample(0), batch_size, repeat_per_collect)
                train_collector.reset_buffer()
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                global_step += step

                for k in result.keys():
                    data[k] = f'{result[k]:.2f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, result[k], global_step=global_step)

                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f'{stat[k].get():.6f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=global_step)
                step_now += step
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)

        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
            if save_fn:
                save_fn(policy)

        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')

        if stop_fn and stop_fn(best_reward):
            break

    return gather_info(
        start_time, train_collector, test_collector, best_reward)
