"""
Stroke class definition.
"""
from __future__ import print_function, division

from .stroke_type import StrokeType
from ..rendering import offset_stk
from ..splines import get_stk_from_bspline


class Stroke(object):
    """
    STROKETOKEN. Random variables that define a continuous pen trajectory
        Utilized in the MotorProgram class

    Reference to StrokeType object "myType"
        This might be shared between multiple strokes (see MotorProgram)
    """
    # tracked properties
    __po = {'pos_token', 'invscales_token', 'shapes_token'}

    def __init__(self, previousStroke=None):
        """
        Initialize the Stroke class instance.

        :param previousStroke: [Stroke] the previous stroke
        """
        if previousStroke is not None:
            assert isinstance(previousStroke, StrokeType)
            self.myType = previousStroke.myType
        else:
            self.myType = StrokeType()

        # token-level parameters
        self.pos_token = None
        self.invscales_token = None
        self.shapes_token = None
        self.eval_spot_token = None

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
        out = self.myType.R
        if (out is not None) and (out.type == 'mid'):
            out.eval_spot_token = self.eval_spot_token

        return out

    @R.setter
    def R(self, val):
        self.myType.R = val
        if (self.myType.R is not None) and (self.myType.R.type == 'mid'):
            self.myType.R.eval_spot_token = []
            self.eval_spot_token = val.eval_spot_token

    @property
    def nsub(self):
        """
        Get the number of sub-strokes
        """
        return len(self.ids)

    @property
    def motor(self):
        """
        Compute the [x,y,t] trajectory of this stroke
        """
        motor = vanilla_to_motor(
            self.shapes_token, self.invscales_token, self.pos_token
        )

        return motor

    @property
    def motor_spline(self):
        """
        TODO
        """
        raise NotImplementedError('motor_spline method not yet implemented.')

def vanilla_to_motor(shapes, invscales, first_pos):
    vanilla_traj = []
    motor = []
    ncpt,_,n = shapes.shape
    for i in range(n):
        shapes[:,:,i] = invscales[i] * shapes[:,:,i]
        vanilla_traj.append(get_stk_from_bspline(shapes[:,:,i]))

        #calculate offset
        if i == 0:
            offset = vanilla_traj[i][0,:] - first_pos
        else:
            offset = vanilla_traj[i-1][0,:] - motor[i-1][-1,:]
        motor.append(offset_stk(vanilla_traj[i],offset))

    return motor
