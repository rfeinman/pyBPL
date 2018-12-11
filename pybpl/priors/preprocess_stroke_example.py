import numpy as np
from scipy.io import loadmat
import Dataset

print("Loading Data...")
data = loadmat('data_background',variable_names=['drawings','images','names','timing'])
D = Dataset.Dataset(data['drawings'],data['images'],data['names'],data['timing'])
print("Data Loaded")

D.make_substroke_dict()