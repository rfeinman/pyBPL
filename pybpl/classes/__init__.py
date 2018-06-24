from .spatial_hist import SpatialHist
from .spatial_model import SpatialModel
from .library import Library
from .relations import (Relation, RelationAttach, RelationAttachAlong,
                        RelationIndependent)
from .stroke import Stroke, StrokeType
from .motor_program import MotorProgram
from .character_type import CharacterType

from . import CPDUnif
from . import CPD
from . import motor_program
from . import spatial_hist
from . import spatial_model
from . import stroke
from . import library
from . import relations
from . import UtilMP

# classes = [
#     'MotorProgram', 'SpatialModel', 'SpatialHist', 'Stroke', 'StrokeType',
#     'Library', 'Relation', 'RelationAttach', 'RelationAttachAlong',
#     'RelationIndependent'
# ]
# modules = [
#     'motor_program', 'spatial_hist', 'spatial_model', 'relations', 'CPD',
#     'CPDUnif', 'UtilMP', 'stroke', 'library'
# ]
# __all__ = classes + modules