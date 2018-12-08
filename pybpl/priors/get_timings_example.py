import numpy as np
from scipy.io import loadmat
import Dataset

print("Loading Data...")
data = loadmat('data_background',variable_names=['drawings','images','names','timing'])
D = Dataset.Dataset(data['drawings'],data['images'],data['names'],data['timing'])
print("Data Loaded")

alphabet=0
character=0
rendition=0
some_drawing = D.drawings[alphabet][character][rendition]
num_strokes = len(some_drawing)

print("Alphabet:",alphabet)
print("Character:",character)
print("Rendition:",rendition)


for s in range(num_strokes):
    print("Stroke:",s)
    stroke = some_drawing[s]
    num_discrete_steps = len(stroke)
    for discrete_step in range(num_discrete_steps):
        x,y,t = stroke[discrete_step]
        print("     ","x:",x,"y:",y,"time:",t)
