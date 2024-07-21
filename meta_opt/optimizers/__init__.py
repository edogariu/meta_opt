from .base import OptimizerConfig
from .schedules import ScheduleConfig, ConstantScheduleConfig, CosineScheduleConfig, LinearScheduleConfig
from .sgd import SGDConfig
from .adamw import AdamWConfig
from .cocob import COCOBConfig
from .dadaptation import DAdaptationConfig
from .dog import DoGConfig
from .dowg import DoWGConfig
from .polyak import PolyakConfig
from .mechanic import MechanicConfig
from .hgd import SGDHGDConfig, AdamHGDConfig

from .metaopt import MetaOptConfig