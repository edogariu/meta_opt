import abc
from typing import Tuple, Dict, Any

from algorithmic_efficiency import spec

from meta_opt.optimizers.base import OptimizerConfig

"""
Defines the API with which we interact with neural networks.
Like `optimizers.py`, this is made to capture multiple deep learning frameworks (jax, pytorch).
"""

class TrainState(metaclass=abc.ABCMeta):
    t: int

    @abc.abstractmethod
    def get_algoperf_stuff(self) -> Tuple[spec.OptimizerState, spec.ParameterContainer, spec.ModelAuxiliaryState]:
        """Returns a tuple of the things needed by `algorithmic_efficiency` for checkpoints and such."""

    @abc.abstractmethod
    def reset(self, 
              rng: spec.RandomState,
              workload: spec.Workload,
              reset_opt_state: bool):
        """Resets model parameters, auxiliary state, and potentially the optimizer state."""

    @abc.abstractmethod
    def get_num_params(self) -> int:
        """Computes the number of trainable parameters."""

    @abc.abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """Computes the number of bytes being used to store the model and optimizer state."""

    @abc.abstractmethod
    def get_logging_metrics(self) -> Dict[str, Any]:
        """Returns a dictionary of metrics important to the train state. 
        Use this to return (for example) things related to the optimizer, such as current learning rate/meta-optimization parameters."""


@abc.abstractmethod
def create_train_state(rng: spec.RandomState,
                       workload: spec.Workload,
                       optimizer_cfg: OptimizerConfig):
    """Creates a train state from scratch. This should initialize model parameters, auxiliary state, and optimizer state."""

@abc.abstractmethod
def load_train_state(checkpoint,
                     workload: spec.Workload,
                     optimizer_cfg: OptimizerConfig):
    """Creates a train state from a checkpoint given by `algorithmic_efficiency`."""

@abc.abstractmethod
def forward(rng: spec.RandomState, 
            workload: spec.Workload,
            tstate: TrainState,
            batch) -> float:
    """Takes a single forward pass, returning only the loss."""

@abc.abstractmethod
def train_step(rng: spec.RandomState, 
               workload: spec.Workload,
               tstate: TrainState,
               batch) -> Tuple[TrainState, Dict[str, Any]]:
    """Takes a single training step, returning the new `tstate` as well as a dictionary of other info such as the loss and the gradients."""
