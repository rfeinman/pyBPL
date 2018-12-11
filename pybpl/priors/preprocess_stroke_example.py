import numpy as np
from scipy.io import loadmat
import Dataset

# D.drawings[alphabet][character][rendition][stroke][step] is an (x,y,time) tuple
# D.substroke_dict[alphabet][character][rendition][stroke][substroke][step] is an (x,y) pair

print("Loading Data...")
data = loadmat('data_background',variable_names=['drawings','images','names','timing'])
D = Dataset.Dataset(data['drawings'],data['images'],data['names'],data['timing'])
print("Data Loaded")

D.make_substroke_dict()