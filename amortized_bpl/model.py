from pybpl.library import Library
from pyprob import Model
from pybpl.model import CharacterModel
import pyprob


class BPL(Model):
    def __init__(self, lib_dir='../lib_data'):
        super().__init__(name="BPL")
        self.lib = Library(lib_dir=lib_dir, use_hist=True)
        self.model = CharacterModel(self.lib)

    def forward(self):
        char_type = self.model.sample_type()
        char_token = self.model.sample_token(char_type)

        image_dist = self.model.image_dist.image_dist(char_token)
        pyprob.observe(image_dist, name='image')
        return char_type, char_token
