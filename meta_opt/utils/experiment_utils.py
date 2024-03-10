from typing import Tuple
from collections import defaultdict
from copy import deepcopy
import dill as pkl
import os
import matplotlib.pyplot as plt
import re
import matplotlib.animation as animation

import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import jax


# some utilities
def _set_seed(seed):
    if seed is None:
        seed = np.random.randint()
        print('seed set to {}'.format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = jax.random.PRNGKey(seed)
    return rng, seed

def get_opt_hyperparams(opt_state):
    """
    helper fn to serialize optax optimizer hyperparameters from the opt_state
    """
    if isinstance(opt_state, Tuple): h = [deepcopy(o.hyperparams) for o in opt_state if hasattr(o, 'hyperparams')]
    else: h = deepcopy(opt_state.hyperparams)
    return h


class bcolors:  # for printing pretty colors :)
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def make(cfg):
    name, directory, load_experiment_checkpoint = cfg['experiment_name'], cfg['directory'], cfg['load_checkpoint']
    print(f'using {bcolors.WARNING}{bcolors.BOLD}{jax.lib.xla_bridge.get_backend().platform}{bcolors.ENDC} for jax')
    print(f'results will be stored at: {bcolors.OKCYAN}{bcolors.BOLD}{directory}/data/{name}_*.pkl{bcolors.ENDC}')
    print('we will {}try to load experiment checkpoint first'.format('' if load_experiment_checkpoint else f'{bcolors.FAIL}{bcolors.BOLD}NOT{bcolors.ENDC} '))
    
    # load checkpoints
    filename = '{}/data/{}_raw.pkl'.format(directory, cfg['experiment_name'])
    exists = os.path.isfile(filename)
    if load_experiment_checkpoint:
        if not exists: 
            print(f'\t{bcolors.FAIL}{bcolors.BOLD}checkpoint could not be found!{bcolors.ENDC}')
            load_experiment_checkpoint = False
        else:
            results = pkl.load(open(filename, 'rb'))
            print(f'\t{bcolors.OKGREEN}{bcolors.BOLD}loaded checkpoint from {filename}, containing {list(results.keys())}{bcolors.ENDC}')
    if not load_experiment_checkpoint:
        results = defaultdict(list)
        if exists:
            print(f'{bcolors.FAIL}{bcolors.BOLD}WARNING: there already exists a checkpoint with this name! make sure you want to overwrite{bcolors.ENDC}')
            assert cfg['overwrite'], 'cannot start from scratch with an existing checkpoint and `overwrite=False`'
        print('starting the experiment from scratch :)')
    if cfg['full_batch']: print(f'{bcolors.FAIL}{bcolors.BOLD}note: using full_batch means we will never eval{bcolors.ENDC}')
    return results

def save_checkpoint(cfg, results, checkpoint_name: str = ''):
    assert len(results) > 0
    filename = '{}/data/{}_raw.pkl'.format(cfg['directory'], cfg['experiment_name'])
    with open(filename, 'wb') as f:
        pkl.dump(results, f)
        print(f'{bcolors.OKBLUE}{bcolors.BOLD}Saved checkpoint {checkpoint_name} to {filename}{bcolors.ENDC}')
        

def process_results(cfg, results):
    # clean the stats
    to_del = []
    for k, v in results.items(): 
        if len(v) == 0: to_del.append(k)
    for k in to_del: del results[k]

    # gather stats
    aggregated = {}  # experiment name -> 'args' or timestamp -> stat key -> stat value
    for k, v in results.items():  # for each experiment
        aggregated[k] = {'args': []}
        for n in range(len(v)):  # for each trial
            aggregated[k]['args'].append(v[n]['args'])
            for t in range(v[0]['args']['num_iters']):  # for each timestamp
                if t not in v[n]: continue
                for stat_key, value in v[n][t].items():  # for each stat recorded at that timestamp
                    if stat_key not in aggregated[k]: aggregated[k][stat_key] = {}
                    if t not in aggregated[k][stat_key]: aggregated[k][stat_key][t] = []
                    aggregated[k][stat_key][t].append(value)

    # aggregate stats
    ret = defaultdict(dict)  # stat key -> experiment name -> 't' or 'avg' or 'std' ->
    args = {}
    for k, v in aggregated.items():  # for experiment
        for stat_key in v.keys():  # for stat
            if stat_key in ['args', 'bleu']:
                args[k] = v[stat_key]
                continue
            if k not in ret[stat_key]: ret[stat_key][k] = {}
            ret[stat_key][k]['t'] = list(v[stat_key].keys())
            arr = np.array(list(v[stat_key].values()))
            ret[stat_key][k]['avg'] = np.mean(arr, axis=1)
            ret[stat_key][k]['std'] = np.std(arr, axis=1)

    filename = '{}/data/{}_processed.pkl'.format(cfg['directory'], cfg['experiment_name'])
    if cfg['overwrite'] or not os.path.isfile(filename): 
        with open(filename, 'wb') as f: pkl.dump(ret, f)
        print(f'{bcolors.OKGREEN}{bcolors.BOLD}Saved processed results to {filename}{bcolors.ENDC}')
    else:
        print(f'{bcolors.FAIL}{bcolors.BOLD}cannot save processed results with existing processed results and `overwrite=False`{bcolors.ENDC}')
    return ret

def get_final_cparams(processed_results, experiment_name: str):
    assert 'M' in processed_results, 'no existing meta experiment'
    p = processed_results['M'][experiment_name]
    assert len(p) > 0, f'{experiment_name} is not a meta experiment'
    return {'M': p['avg'][-1]}


def animate(results, Ms, downsample, bounds):
    downsample_factor = downsample  # how many timesteps to move forward every animation step
    ymin, ymax = bounds

    anim_data = []  # each entry is a dictionary containing the M values for that animation step
    _Ms = {k: (np.array(v[0]), v[1]) for k, v in Ms.items()}
    H_max = max([v[1].shape[1] for v in _Ms.values()])
    T = list(results.values())[0][0]['args']['num_iters']
    name = results['experiment_name']
    for t in range(0, T, downsample_factor):
        temp = {}
        for k, (ts, vals) in _Ms.items(): temp[k] = vals[max(0, np.argmax(ts > t) - 1)]
        anim_data.append(temp)

    fig = plt.figure()  # initializing a figure in which the graph will be plotted
    ax = plt.axes(xlim =(0, H_max), ylim=(ymin, ymax))  # marking the x-axis and y-axis
    ax.set_xlabel('number of steps in the past')
    ax.set_ylabel('M coefficient')

    # initializing a line variable
    ls = {}
    for k in _Ms.keys():
        ls[k], = ax.plot([], [], lw = 3, label=k)
    legend = ax.legend()

    # data which the line will contain (x, y)
    def init():
        for l in ls.values(): l.set_data([], [])
        return list(ls.values())

    def animate(i):
        for k, M in anim_data[i].items():
            x, y = range(0, len(M)), M
            ls[k].set_data(x, y[::-1])
            # line.set_label(i)
        # legend.get_texts()[0].set_text(i * downsample_factor) #Update label each at frame
        ax.set_title(f'timestep #{i * downsample_factor} of meta-opt on {name}')
        return list(ls.values())

    anim = animation.FuncAnimation(fig, animate, init_func = init,
                        frames = T // downsample_factor, interval = downsample_factor, blit = True)
    return anim

def plot(results, processed_results, keys_to_plot, anim_downsample_factor=200, anim_bounds=(-0.4, 0.1)):
    fig, ax = plt.subplots(len(processed_results), 1, figsize=(10, 32))
    Ms = {}
    for i, stat_key in enumerate(processed_results.keys()):
        ax[i].set_title(stat_key)
        for experiment_name in processed_results[stat_key].keys():
            if (isinstance(keys_to_plot, list) and experiment_name not in keys_to_plot) or (isinstance(keys_to_plot, str) and not re.match(keys_to_plot, experiment_name)): 
                # print(f'skipped {experiment_name}')
                continue
            ts, avgs, stds = processed_results[stat_key][experiment_name]['t'], processed_results[stat_key][experiment_name]['avg'], processed_results[stat_key][experiment_name]['std']
            if avgs.ndim == 2:  # how to handle stats that are vectors (such as the Ms for scalar meta-opt)
                Ms[experiment_name] = (ts, avgs, stds)
                
                _t, _a, _s = range(avgs.shape[1]), avgs[-1][::-1], stds[-1][::-1]
                ax[i].plot(_t, _a, label=experiment_name)
                ax[i].fill_between(_t, _a - 1.96 * _s, _a + 1.96 * _s, alpha=0.2)
                ax[i]
                
                # ax[i].plot(ts, avgs.sum(axis=-1), label=experiment_name)
                # stds = ((stds ** 2).sum(axis=-1)) ** 0.5
                # ax[i].fill_between(ts, avgs.sum(axis=-1) - 1.96 * stds, avgs.sum(axis=-1) + 1.96 * stds, alpha=0.2)
                
                # for j in range(avgs.shape[1]):
                #     ax[i].plot(ts, avgs[:, j], label=f'{experiment_name} {str(j)}')
                #     ax[i].fill_between(ts, avgs[:, j] - 1.96 * stds[:, j], avgs[:, j] + 1.96 * stds[:, j], alpha=0.2)
            else:
                if stat_key in ['loss', 'grad_sq_norm', 'eval_acc', 'eval_loss']:
                    n = 4
                    kernel = np.array([1 / n,] * n)
                    avgs = np.convolve(avgs, kernel)[n // 2:n // 2 + avgs.shape[0]]
                    stds = np.convolve(stds ** 2, kernel ** 2)[n // 2:n // 2 + stds.shape[0]] ** 0.5
                ax[i].plot(ts, avgs, label=experiment_name)
                ax[i].fill_between(ts, avgs - 1.96 * stds, avgs + 1.96 * stds, alpha=0.2)
    for a in ax: a.legend()
    anim = animate(results, Ms, anim_downsample_factor, anim_bounds)
    plt.close()
    return (fig, ax), anim
