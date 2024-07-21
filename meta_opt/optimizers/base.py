import abc
import optax

# from typing import Callable, Iterable
# from torch import optim, Tensor

"""
Optimizer hyperparameters and meta-information in a *makeable* config format.
Each optimizer config will have a way to make that optimizer for torch or for jax, see
below for details
"""
OPT_REGISTRY = {}  # name -> class. Gets populated by calls to register()

class OptimizerConfig(abc.ABC):
    optimizer_name: str
    self_tuning: bool
    reset_opt_state: bool

    @staticmethod
    def from_dict(d: dict):
        """dict -> config

        Args:
            d (dict): dict assumed to contain all of the required args for the corresponding optimizer

        Returns:
            OptimizerConfig: config made from the dict
        """
        assert 'optimizer_name' in d
        name = d['optimizer_name']
        assert name in OPT_REGISTRY, f'no known optimizer named {name}'
        cls = OPT_REGISTRY[name]
        assert cls.optimizer_name == name, f'class {cls} has wrong optimizer name, got {cls.optimizer_name} and expected {name}'
        return cls.from_dict(d)
    
    @staticmethod
    def register(opt_cls):
        """Registers an optimizer config class with the registry!"""
        assert hasattr(opt_cls, 'optimizer_name'), 'optimizer class must have an `optimizer_name` attribute!'
        assert hasattr(opt_cls, 'from_dict'), 'optimizer class must have a `from_dict` method!'
        opt_name = opt_cls.optimizer_name
        assert opt_name not in OPT_REGISTRY, f'optimizer {opt_name} already registered'
        OPT_REGISTRY[opt_name] = opt_cls
        return opt_cls

    @abc.abstractmethod
    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """

    # @abc.abstractmethod
    # def make_torch(self) -> Callable[[Iterable[Tensor]], optim.Optimizer]:
    #     """
    #     Instantiates this optimizer configuration for use with pytorch. 
    #     For example, if this were SGD, it would return roughly the same thing as
    #             `lambda params: torch.optim.SGD(params, lr=self.lr, ...)`
    #     and could be used afterward in the usual way.
    #     """

    def make(self, framework: str):
        if framework == 'torch': 
            raise NotImplementedError('havent gotten around to implementing torch yet, sorry')
            # return self.make_torch()
        elif framework == 'jax':
            return self.make_jax()
        else:
            raise NotImplementedError(framework)
