import abc
from typing import Optional

import chex
import optax

"""
Schedule hyperparameters and meta-information in a *makeable* config format.
Returns a function that takes the current iteration and returns the learning rate.
"""

SCHEDULER_REGISTRY = {}  # name -> class. Gets populated by calls to register()

class ScheduleConfig(abc.ABC):
    scheduler_name: str

    @staticmethod
    def from_dict(d: dict):
        assert 'scheduler_name' in d
        name = d['scheduler_name']
        assert name in SCHEDULER_REGISTRY, f'no known scheduler named {name}'
        cls = SCHEDULER_REGISTRY[name]
        assert cls.scheduler_name == name, f'class {cls} has wrong scheduler name, got {cls.scheduler_name} and expected {name}'
        return cls(**d)
    
    @staticmethod
    def register(scheduler_cls):
        """Registers a scheduler config class with the registry!"""
        assert hasattr(scheduler_cls, 'scheduler_name'), 'optimizer class must have a `scheduler_name` attribute!'
        scheduler_name = scheduler_cls.scheduler_name
        assert scheduler_name not in SCHEDULER_REGISTRY, f'scheduler {scheduler_name} already registered'
        SCHEDULER_REGISTRY[scheduler_name] = scheduler_cls
        return scheduler_cls

    @abc.abstractmethod
    def make_jax(self) -> optax.Schedule:
        """
        Instantiates this lr schedule configuration for use with jax/flax/optax. 
        For example, if this were a constant schedule, it would return roughly the same thing as
                `optax.constant_schedule(value=self.lr, ...)`
        and could be used afterward in the usual way.
        """

@ScheduleConfig.register
@chex.dataclass
class ConstantScheduleConfig(ScheduleConfig):
    # REQUIRED
    learning_rate: float

    # METADATA
    scheduler_name: str = 'Constant'

    def make_jax(self) -> optax.Schedule:
        return optax.constant_schedule(value=self.learning_rate)

@ScheduleConfig.register
@chex.dataclass
class CosineScheduleConfig(ScheduleConfig):
    # REQUIRED
    init_value: float
    decay_steps: int

    # OPTIONAL
    peak_value: Optional[float] = None
    warmup_steps: Optional[int] = None
    alpha: float = 1e-5  # a minimum value to ensure LR doesnt drop to 0
    exponent: float = 1.0  # exponent for cosine decay

    # METADATA
    scheduler_name: str = 'Cosine'

    def make_jax(self) -> optax.Schedule:
        assert (self.peak_value is None) == (self.warmup_steps is None), 'peak value and warmup steps must both be None or both not None'
        if self.peak_value is None:
            return optax.cosine_decay_schedule(init_value=self.init_value, 
                                            decay_steps=self.decay_steps, 
                                            alpha=self.alpha, 
                                            exponent=self.exponent)
        else:
            return optax.warmup_cosine_decay_schedule(init_value=self.init_value, 
                                            peak_value=self.peak_value, 
                                            warmup_steps=self.warmup_steps, 
                                            decay_steps=self.decay_steps, 
                                            end_value=self.alpha, 
                                            exponent=self.exponent)

@ScheduleConfig.register
@chex.dataclass
class LinearScheduleConfig(ScheduleConfig):
    # REQUIRED
    init_value: float
    end_value: float
    decay_steps: int

    # METADATA
    scheduler_name: str = 'Linear'

    def make_jax(self) -> optax.Schedule:
        return optax.linear_schedule(init_value=self.init_value,
                                     end_value=self.end_value,
                                     transition_steps=self.decay_steps)
    
