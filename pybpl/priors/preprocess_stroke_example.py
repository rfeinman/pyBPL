'''
NOTE: Dataset.partition_stroke function is not translated yet.
Once that's done, this is how we pre-process the raw stroke data
before finding primatives and so on
'''

import numpy as np
from scipy.io import loadmat
import Dataset
from scipy.interpolate import interp1d

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


'''
Example on one sub-stroke:
    - convert into unif time
    - break into substrokes
    - convert each substroke into unif space

'''
stroke = some_drawing[0]
stroke,unif_time = Dataset.unif_time(stroke)
substrokes,_,_ = Dataset.partition_stroke(stroke) # THIS FUNCTION IS NOT YET DONE
first_substroke = substrokes[0]
first_substroke = Dataset.unif_space(first_substroke)