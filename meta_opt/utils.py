from typing import Tuple
import sys
import pprint
from absl import logging

import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# global mesh for sharding
GLOBAL_MESH: Mesh = None

# set up pretty printer
pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def enable(_):
        bcolors.HEADER = '\033[95m'
        bcolors.OKBLUE = '\033[94m'
        bcolors.OKCYAN = '\033[96m'
        bcolors.OKGREEN = '\033[92m'
        bcolors.WARNING = '\033[93m'
        bcolors.FAIL = '\033[91m'
        bcolors.ENDC = '\033[0m'
        bcolors.BOLD = '\033[1m'
        bcolors.UNDERLINE = '\033[4m'
    
    @classmethod
    def disable(_):
        bcolors.HEADER = bcolors.OKBLUE = bcolors.OKCYAN = bcolors.OKGREEN = bcolors.WARNING = bcolors.FAIL = bcolors.ENDC = bcolors.BOLD = bcolors.UNDERLINE = ''

def pretty_dict(obj):
    pretty_out = f"{pp.pformat(obj)}"
    return f'{pretty_out}'

def get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size

def make_mesh(batch_num_devices: int, opt_state_num_devices: int) -> Mesh:
    global GLOBAL_MESH
    assert batch_num_devices * opt_state_num_devices == jax.local_device_count(), (batch_num_devices, opt_state_num_devices, jax.local_device_count())

    devices = mesh_utils.create_device_mesh((batch_num_devices, opt_state_num_devices))
    GLOBAL_MESH = Mesh(devices, axis_names=('batch', 'opt'))
    assert GLOBAL_MESH.shape == {'batch': batch_num_devices, 'opt': opt_state_num_devices}
    n_devices = jax.local_device_count()
    logging.info(f' {bcolors.WARNING}{bcolors.BOLD}{n_devices} devices in a mesh {GLOBAL_MESH} of shape {GLOBAL_MESH.shape}{bcolors.ENDC}')
    return GLOBAL_MESH

def get_mesh() -> Mesh:
    global GLOBAL_MESH
    assert GLOBAL_MESH is not None, 'didnt set up mesh yet!'
    assert 'batch' in GLOBAL_MESH.axis_names and 'opt' in GLOBAL_MESH.axis_names
    return GLOBAL_MESH

def sharding_constraint(arr: jax.Array, spec: Tuple[str]) -> jax.Array:
    s = NamedSharding(get_mesh(), P(*spec))
    return jax.lax.with_sharding_constraint(arr, s)

def shard(arr: jax.Array, spec: Tuple[str]) -> jax.Array:
    s = NamedSharding(get_mesh(), P(*spec))
    return jax.device_put(arr, s)
