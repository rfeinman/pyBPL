from .motor_program import MotorProgram
from .spatial_hist import SpatialHist
from .spatial_model import SpatialModel
from .stroke import Stroke

from . import motor_program
from . import spatial_hist
from . import spatial_model
from . import stroke

__all__ = ['MotorProgram', 'SpatialModel', 'SpatialHist', 'Stroke',
           'motor_program', 'spatial_hist', 'spatial_model', 'stroke']