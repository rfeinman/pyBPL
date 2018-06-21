"""
Stroke class definition.
"""
from __future__ import print_function, division
import copy

from .stroke_type import StrokeType
from .. import rendering


class Stroke(object):
    """
    STROKETOKEN. Random variables that define a continuous pen trajectory
        Utilized in the MotorProgram class

    Reference to StrokeType object "myType"
        This might be shared between multiple strokes (see MotorProgram)
    """
    # tracked properties
    po = {'pos_token', 'invscales_token', 'shapes_token'}

    def __init__(self, previousStroke=None):
        """
        Initialize the Stroke class instance.

        :param previousStroke: [Stroke] the previous stroke
        """
        self.refresh_listener()
        if previousStroke is not None:
            # TODO - verify that copying works as expected
            self.myType = copy.deepcopy(previousStroke.myType)
        else:
            self.myType = StrokeType()

        # token-level parameters
        self.pos_token = []
        self.invscales_token = []
        self.shapes_token = []

        self.lh = None
        self.cache_current = False

    @property
    def ids(self):
        return self.myType.ids

    @property
    def invscales_type(self):
        return self.myType.invscales_type

    @property
    def shapes_type(self):
        return self.myType.shapes_type

    @property
    def R(self):
        out = self.myType.R
        if (out is not None) and (out.type == 'mid'):
            out.eval_spot_token = self.eval_spot_token

        return out

    def set_ids(self, val):
        self.myType.ids = val

    def set_invscales_type(self, val):
        self.myType.invscales_type = val

    def set_shapes_type(self, val):
        self.myType.shapes_type = val

    def set_R(self, val):
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
        Compute the [x,y,t] trajectory of this stroke, either from cached item
        or from scratch
        """
        assert self.onListener
        if self.cache_current:
            motor = self.cache_motor
        else:
            motor = vanilla_to_motor(
                self.shapes_token, self.invscales_token, self.pos_token
            )
            self.cache_motor = motor
            self.cache_current = True

        return motor

    @property
    def motor_spline(self):
        """
        TODO
        """
        raise NotImplementedError('motor_spline method not yet implemented.')

    def saveobj(self):
        Y = copy.deepcopy(self)
        del Y.lh

    def onListener(self):
        return True

    def refresh_listener(self):
        return

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
        motor.append(rendering.offset_stk(vanilla_traj[i],offset))

    return motor
