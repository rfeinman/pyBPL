from __future__ import division, print_function
import unittest

from ..pybpl import CPD
from ..pybpl.library.library import Library
from ..pybpl import rendering

class TestSpatialHist(unittest.TestCase):

    def setUp(self):
        self.lib = Library('../lib_data')

    def testVanillaToMotor(self):
        # get shapes and invscales token
        stroke = CPD.sample_stroke_type(self.lib, ns=1)
        st = stroke.sample_token()
        shapes_token = st.shapes
        invscales_token = st.invscales
        # get position
        relation = CPD.sample_relation_type(self.lib, [])
        position = relation.sample_position([])
        # call vanilla_to_motor
        motor, _ = rendering.vanilla_to_motor(
            shapes_token, invscales_token, position
        )
        print(motor)