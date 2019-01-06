from scipy.io import loadmat
from matplotlib import pyplot as plt

from ..dataset import Dataset
from . import flows

print("Loading Data")
data = loadmat('data_background',variable_names=['drawings','images','names','timing'])
D = Dataset(data['drawings'],data['images'],data['names'],data['timing'])
first_strokes = D.first_stroke_locations()
print("Data Loaded")


plt.scatter(first_strokes[:,0],first_strokes[:,1])
plt.title("First Stroke Locations")
plt.show()

model = flows.FlowDensityEstimator(first_strokes, 
                                    num_blocks=4,
                                    num_hidden=64,
                                    lr=.0001,
                                    batch_size=100,
                                    test_batch_size=1000,
                                    log_interval=1000,
                                    seed=1)

model.fit(epochs=1000)

model.plot()
