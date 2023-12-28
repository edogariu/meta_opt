import optax
from flax import struct

class ControllerState(struct.PyTreeNode):
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    
# TODO 