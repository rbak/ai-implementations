from .reinforcement_learning import *
from .deep_learning import *
from .deep_rl import *
from .base import BaseAlgorithm

__all__ = [cls.__name__ for cls in BaseAlgorithm.__subclasses__()]
