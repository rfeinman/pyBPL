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

    # Unif Time
    stroke = Dataset.unif_time(stroke)
    num_discrete_steps = len(stroke)
    for discrete_step in range(num_discrete_steps):
        x,y,t = stroke[discrete_step]
        print("(unif)     ","x:",x,"y:",y,"time:",t)

    # Substroke

    ''' # Not done
    stroke = Dataset.partition_strokes(stroke)
    num_discrete_steps = len(stroke)
    for discrete_step in range(num_discrete_steps):
        x,y,t = stroke[discrete_step]
        print("(unif)     ","x:",x,"y:",y,"time:",t)
    '''

    # Unif Space
    ''' # Almost done
    stroke = Dataset.unif_space(stroke)
    num_substrokes = len(stroke)
    for idx,substroke in enumerate(num_substrokes):
        print("substroke #:",idx)
        num_discrete_steps = len(substroke)
        for discrete_step in range(num_discrete_steps):
            print("x:",x,"y:",y,"time:",t)
    '''


