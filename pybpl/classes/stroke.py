"""
Stroke class definition.
"""
from __future__ import print_function, division

from ..rendering import offset_stk
from ..splines import get_stk_from_bspline


class StrokeType(object):
    def __init__(self, ids=None, shapes_type=None, invscales_type=None):
        self.ids = ids
        self.shapes_type = shapes_type
        self.invscales_type = invscales_type

    @property
    def nsub(self):
        return len(self.ids)

class Stroke(object):
    """
    STROKETOKEN. Random variables that define a continuous pen trajectory
        Utilized in the MotorProgram class

    Reference to StrokeType object "myType"
        This might be shared between multiple strokes (see MotorProgram)
    """

    def __init__(
            self, stype=None, pos_token=None, invscales_token=None,
            shapes_token=None
    ):
        """
        Initialize the Stroke class instance.

        :param stype: [StrokeType] the type-level template for the stroke
        """
        if stype is not None:
            assert isinstance(stype, StrokeType)
            self.myType = stype
        else:
            self.myType = StrokeType()

        # token-level parameters
        self.pos_token = pos_token
        self.invscales_token = invscales_token
        self.shapes_token = shapes_token

    # -----
    # NOTE: these might be unnecessary
    # -----
    @property
    def ids(self):
        return self.myType.ids

    @ids.setter
    def ids(self, val):
        self.myType.ids = val

    @property
    def invscales_type(self):
        return self.myType.invscales_type

    @invscales_type.setter
    def invscales_type(self, val):
        self.myType.invscales_type = val

    @property
    def shapes_type(self):
        return self.myType.shapes_type

    @shapes_type.setter
    def shapes_type(self, val):
        self.myType.shapes_type = val

    @property
    def R(self):
        return self.myType.R

    @R.setter
    def R(self, val):
        self.myType.R = val

    @property
    def nsub(self):
        """
        Get the number of sub-strokes
        """
        return self.myType.nsub

    @property
    def motor(self):
        """
        Compute the [x,y,t] trajectory of this stroke
        """
        motor, _ = vanilla_to_motor(
            self.shapes_token, self.invscales_token, self.pos_token
        )

        return motor

    @property
    def motor_spline(self):
        """
        Compute the spline trajectory of this stroke
        """
        raise NotImplementedError
        _, motor_spline = vanilla_to_motor(
            self.shapes_token, self.invscales_token, self.pos_token
        )

def vanilla_to_motor(shapes, invscales, first_pos):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with k sub-strokes

    :param shapes: [(ncpt,2,k) tensor] spline points in normalized space
    :param invscales: [(k,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :return:
        motor: [list] k-length fine motor sequence
        motor_spline: [list] k-length fine motor sequence in spline space
    """
    vanilla_traj = []
    motor = []
    ncpt,_,n = shapes.shape
    for i in range(n):
        shapes[:,:,i] = invscales[i] * shapes[:,:,i]
        vanilla_traj.append(get_stk_from_bspline(shapes[:,:,i]))

        # calculate offset
        if i == 0:
            offset = vanilla_traj[i][0,:] - first_pos
        else:
            offset = vanilla_traj[i-1][0,:] - motor[i-1][-1,:]
        motor.append(offset_stk(vanilla_traj[i],offset))

    motor_spline = None

    return motor, motor_spline
