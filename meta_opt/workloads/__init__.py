import os
from copy import deepcopy
import jax
import optax

from meta_opt.nn import create_train_state
from meta_opt.utils.experiment_utils import _set_seed, bcolors

from .nonconvex_quadratic import load_ncq, NCQ
from .mnist import load_mnist, MLP, CNN
from .cifar10 import load_cifar10, VGG16
from .wmt import load_wmt, WMT

def get_workload(cfg, optimizer):
    rng, cfg['seed'] = _set_seed(cfg['seed'])
    init_rng, rng = jax.random.split(rng)
    directory = cfg['directory']
    
    # get dataset and model
    if cfg['workload'] == 'NONCONVEX_QUADRATIC':
        train_ds, test_ds, example_input, loss_fn, metric_fns = load_ncq(cfg)
        dim = 64
        A = jax.random.normal(jax.random.PRNGKey(cfg['seed']), (dim, dim))
        model = NCQ(dim=dim, std=1e-5, A=A, radius=1.0)
        model.radius = 1.0
    elif cfg['workload'] == 'MNIST':
        if 'model' in cfg and cfg['model'] == 'tiny': 
            print("DOING THE TINY MODEL")
            model = MLP([28 * 28, 10])
        else: model = MLP([28 * 28, 100, 100, 10])
        train_ds, test_ds, example_input, loss_fn, metric_fns = load_mnist(cfg, dataset_dir=os.path.join(directory, 'datasets'))
    elif cfg['workload'] == 'CIFAR':
        train_ds, test_ds, example_input, loss_fn, metric_fns = load_cifar10(cfg, dataset_dir=os.path.join(directory, 'datasets'))
        model = VGG16()
    elif cfg['workload'] == 'WMT':
        train_ds, test_ds, example_input, loss_fn, metric_fns, tokenizer = load_wmt(cfg, dataset_dir=os.path.join(directory, 'datasets'))
        model = WMT(cfg, tokenizer, size=cfg['transformer_size'])
    else:
        raise NotImplementedError(cfg['workload'])

    tstate = create_train_state(init_rng, model, example_input, optimizer, loss_fn, metric_fns=metric_fns)
    del init_rng

    args = deepcopy(cfg)
    if cfg['full_batch']: args['eval_every'] = int(1e9)
        
    args.update({'model': str(model), 
                 'num_params': sum(x.size for x in jax.tree_util.tree_leaves(tstate.params))})
    print(args['num_params'], 'params in the model!')

    return tstate, train_ds, test_ds, rng, args
