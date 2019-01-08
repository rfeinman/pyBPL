import numpy as np
from scipy.interpolate import interp1d
import torch

from ... import splines
from ..primitives.primitive_classifier import PrimitiveClassifierSingle


class Dataset(object):
    def __init__(self,drawings,images,names,timing):
        
        '''
        USAGE:
        import numpy as np
        from scipy.io import loadmat
        data = loadmat('data_background',variable_names=['drawings','images','names','timing'])
        D = Dataset(data['drawings'],data['images'],data['names'],data['timing'])
        
        Images are 105x105 are retrieved with:
        D.images[alphabet][character][rendition] 

        Drawings of a particular image are lists of strokes, where each
        stroke is a list of timesteps, where each timestep is an (x,y) coord.
        (x,y) coords of drawings are retrieved with:
        D.drawings[alphabet][character][rendition][stroke][timestep]
        
        NOTES: 
        (1) ONLY CLEANING UP DRAWINGS AND IMAGES FOR NOW.
            NAMES AND TIMING HAVE UNSPECIFIED INDEX STURCTURE FOR NOW, NOT SAVING THEM IN CLASS
        
        (2) ASK BRENDEN ABOUT FLIP_IMG
        '''

        self.names = names
        self.substroke_dict = None # need to call self.make_substroke_dict()
        self.spline_dict = None # need to call self.make_spline_dict()
        self.subid_dict = None # need to call self.make_subid_dict()

        im_0 = images[0][0][0][0][0][0]
        assert im_0.shape==(105,105)
        self.flip_img = not self.check_black_is_true(im_0)
        
            
        self.images = {}
        self.drawings = {}
        
        n_alpha = len(images)
        for a in range(n_alpha):
            
            alphabet = images[a][0]
            n_char = len(alphabet)
            
            self.images[a] = {}
            self.drawings[a] = {}
            
            for c in range(n_char):
                
                character = alphabet[c][0]
                n_rend = len(character)
                
                self.images[a][c] = {}
                self.drawings[a][c] = {}
                
                for r in range(n_rend):
                    
                    rendition_image = character[r][0]
                    rendition_drawing = drawings[a][0][c][0][r][0]
                    num_strokes = len(rendition_drawing)
                    
                    self.images[a][c][r] = rendition_image
                    
                    # NOTE: ASK BRENDEN ABOUT THIS
                    if self.flip_img:
                        self.images[a][c][r] = (1 - self.images[a][c][r])
                    
    
                    self.drawings[a][c][r] = {}
                    
                    for s in range(num_strokes):
                        
                        stroke = rendition_drawing[s][0]
                        discrete_steps = len(stroke)
                        times = timing[a][0][c][0][r][0][s][0][0]
                        
                        self.drawings[a][c][r][s] = np.zeros((discrete_steps,3))
                
                        for discrete_step in range(discrete_steps):
                            assert len(stroke)==len(times)
                            x,y = stroke[discrete_step]
                            t = times[discrete_step]
                            self.drawings[a][c][r][s][discrete_step] = [x,y,t]

        #D.drawings[alphabet][character][rendition][stroke][step] is an (x,y,time) tuple

    def check_black_is_true(self,I):
        '''
        NOTE: ASK BRENDEN ABOUT THIS
        
        Check whether the black pixels are labeled as
        "true" in the image format, since there should be fewer
        black pixels
        '''
        return ((I==True).sum() < (I==False).sum())

    def first_stroke_locations(self):
        first_strokes = []
        n_alpha = len(self.drawings)
        for a in range(n_alpha):
            alphabet = self.drawings[a]
            n_char = len(alphabet)
            for c in range(n_char):
                char = alphabet[c]
                n_rend = len(char)
                for r in range(n_rend):
                    rendition = char[r]
                    stroke=0
                    first_stroke = rendition[stroke]
                    discrete_step = 0
                    x,y,t = first_stroke[discrete_step]
                    first_stroke_start = [x,y]
                    first_strokes.append(first_stroke_start)
        self.first_strokes = np.vstack(first_strokes)

        return self.first_strokes

    def make_substroke_dict(self):
        print("Making sub-stroke dictionary...")
        substroke_dict = {}
        n_alpha = len(self.drawings)
        for a in range(n_alpha):
            substroke_dict[a] = {}
            alphabet = self.drawings[a]
            n_char = len(alphabet)
            for c in range(n_char):
                substroke_dict[a][c] = {}
                char = alphabet[c]
                n_rend = len(char)
                for r in range(n_rend):
                    substroke_dict[a][c][r] = {}
                    rendition = char[r]
                    n_stroke = len(rendition)
                    for s in range(n_stroke):
                        substroke_dict[a][c][r][s] = {}
                        stroke = self.drawings[a][c][r][s]
                        unif_stroke,unif_stroke_times = unif_time(stroke)
                        substrokes,modified_input,breaks = partition_stroke(unif_stroke)
                        n_substroke = len(substrokes)
                        for ss in range(n_substroke):
                            substroke = substrokes[ss]
                            unif_space_substroke = unif_space(substroke)
                            substroke_dict[a][c][r][s][ss] = unif_space_substroke
        self.substroke_dict = substroke_dict

        return self.substroke_dict
        # D.substroke_dict[alphabet][character][rendition][stroke][substroke][step] is an (x,y) pair

    def make_spline_dict(self):
        assert self.substroke_dict is not None
        ss_dict = self.substroke_dict
        print("Converting sub-strokes to splines...")
        spline_dict = {}
        n_alpha = len(ss_dict)
        for a in range(n_alpha):
            spline_dict[a] = {}
            alphabet = ss_dict[a]
            n_char = len(alphabet)
            for c in range(n_char):
                spline_dict[a][c] = {}
                char = alphabet[c]
                n_rend = len(char)
                for r in range(n_rend):
                    spline_dict[a][c][r] = {}
                    rendition = char[r]
                    n_stroke = len(rendition)
                    for s in range(n_stroke):
                        spline_dict[a][c][r][s] = {}
                        stroke = rendition[s]
                        n_substrokes = len(stroke)
                        for ss in range(n_substrokes):
                            num_steps = len(stroke[ss])
                            if num_steps >= 10:
                                spline_dict[a][c][r][s][ss] = np.zeros((5, 2))
                                substk = stroke[ss]
                                substk, _, scale = norm_substk(substk)
                                spline = splines.fit_bspline_to_traj(substk,
                                                                     nland=5)
                                # PyTorch -> Numpy
                                spline = spline.numpy()
                                # Add 2 extra dimensions - scales weighted twice
                                spline = np.append(spline, [[scale, scale]],
                                                   axis=0)
                                spline_dict[a][c][r][s][ss] = spline
        self.spline_dict = spline_dict

        return self.spline_dict

    def make_subid_dict(self):
        assert self.spline_dict is not None
        spline_dict = self.spline_dict
        clf = PrimitiveClassifierSingle()
        subid_dict = {}
        for a in spline_dict.keys():
            subid_dict[a] = {}
            for c in spline_dict[a].keys():
                subid_dict[a][c] = {}
                for r in spline_dict[a][c].keys():
                    subid_dict[a][c][r] = {}
                    for s in spline_dict[a][c][r].keys():
                        ids = []
                        for ss in spline_dict[a][c][r][s].keys():
                            spline = torch.tensor(
                                spline_dict[a][c][r][s][ss],
                                dtype=torch.float32
                            )
                            prim_ID = clf.predict(spline)
                            ids.append(prim_ID)
                        subid_dict[a][c][r][s] = ids
        self.subid_dict = subid_dict

        return self.subid_dict


def norm_substk(substroke, newscale=105):
    mu = np.mean(substroke, axis=0)
    substroke = substroke - mu
    range_x, range_y = np.ptp(substroke, axis=0)
    scale = newscale / max(1, max(range_x, range_y))
    substroke = substroke * scale

    return substroke, mu, scale

def unif_time(stroke,time_int=50.0):

    '''
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_time_lerp.m
    
 
    Converts a stroke [x,y,t] so that it is uniformly sampled in time

    Inputs:
        - stroke
        - time_int: interpolate stroke so that we have datapoints
                    every interval of this many milliseconds

    Outputs:
        - unif_stroke [k x 2] new stroke
        - unif_time [k x 1] uniform time interval

    '''
    
    times = stroke[:,2]
    min_time = min(times)
    max_time = max(times)
    
    # range excludes endpoint
    unif_times = list(np.arange(min_time,max_time,time_int))
    unif_times.append(max_time)
    unif_times = np.array(unif_times)    
    unif_steps = len(unif_times)
    unif_stroke = np.zeros((unif_steps,2))
    
    for t in range(unif_steps):
                        
        new_time = unif_times[t]
        diffs = times - new_time
    
        # If some thing/things happen at this time
        # just average their values
        # Note, probably only 1 thing happens at this time
        if np.any(diffs==0):
            
            # if some indices have the same time as this new time,
            # average their x,y together
            matches = [xyt for i,xyt in enumerate(stroke) if diffs[i] == 0]
            unif_stroke[t] = np.mean(matches,axis=0)[:2]
          
        # Otherwise interpolate
        else: 

            # last index with time less than new_times[i]
            prev_idx = lt = np.where(diffs<0)[0][-1]

            # first index with time greater than new_times[i]
            post_idx = np.where(diffs>0)[0][0]

            prev_xy = stroke[prev_idx,:2]
            prev_time = times[prev_idx]
            post_xy = stroke[post_idx,:2]
            post_time = times[post_idx]

            # interpolate
            frac = (new_time - prev_time) / (post_time - prev_time)
            assert frac <= 1

            unif_stroke[t] = (1-frac) * prev_xy + frac * post_xy
            
    return unif_stroke,unif_times



# NOTE: assumes dt is always 1
# because vector 1:n is given
# for times in original code
def get_deriv(X):
    steps,dim = X.shape

    dxdt = np.zeros(X.shape)
    
    for i in range(2,steps):
        prev = X[i-1,:]
        _next = X[i,:]
        dxdt[i,:] = _next-prev
        # dt is always 1
    return dxdt

def partition_stroke(stroke,dthresh=1,max_sequence=np.infty):

    '''
    Translated from https://github.com/brendenlake/BPL/blob/master/data/partition_strokes.m
    

    Partition a stroke into sub-strokes based on pauses of the pen
    
    inputs:
        - stroke [n x 2] assuming uniform time sampling (called unif_stk)
        - dthresh: if this much distance/norm is not covered
          at each time point, then it is a pause
        - max_seq: max length of a stop seqeunce, before it
          it is considered its own stroke

    outputs:
        - substrokes: dictionary of length num_substrokes
        - input stroke with pause sequences shorted to a single point (called unif_stk)
        - breaks: where pauses occured
    '''

    # compute derivatives
    modified_input_stroke = np.copy(stroke)
    n = len(stroke)
    dxdt = get_deriv(stroke)

    # Special case
    if n<=2: # MARK: changed from 1 to 2, one seq of length 2 was breaking things...
        substrokes = {}
        substrokes[0] = stroke
        return substrokes,stroke,True
    
    
    # compute norm of derivs
    norm_dxdt = np.zeros((n,1))
    for i in range(n):
        norm_dxdt[i] = np.linalg.norm(dxdt[i])

    # compute candidate stop points
    stop_pt = norm_dxdt < dthresh
    for i in range(1,n):
        if stop_pt[i]:
            stop_pt[i-1] = True 
    stop_pt[0] = True
    stop_pt[-1] = True
    
    '''
    Partition stop points in to sequences
    Here, non-stops are denoted as zeros, the
    first stop is a sequence of 1s, second is a sequence
    2s, etc.. Until the pen is moving fast enough again
    '''
   
    stop_sequence = np.zeros((n,1))
    stop_count = 1
    
    for i in range(n-1):
        if stop_pt[i]: #current point is a stop, it's the same stop
            stop_sequence[i] = stop_count
        elif stop_pt[i-1] and stop_pt[i+1]: 
            # points surround it are a stop... its the same stop
            stop_sequence[i] = stop_count
        elif stop_pt[i-1]:
            stop_count += 1 # just finished a stop
        

    # Special case where the entire stroke is a stop sequence
    if stop_count == 1:
        stop_sequence = np.zeros((n,1))
        stop_sequence[0] = 1
        stop_sequence[-1] = 2
        stop_count = 2
         

    # Make sure the stop sequences aren't too long. If they are,
    # we place a sub-stroke break at the beginning and end.
    i = 0
    while i < stop_count:
        
        sel = [idx for idx,b in enumerate(stop_sequence == i) if b==True]
        nsel = len(sel)
        if nsel>max_sequence:
            stop_sequence[sel[1:-1]] = 0
            idxs = [idx for idx,b in enumerate(stop_sequence > i) if b==True]    
            stop_sequence[idxs] = stop_sequence[idxs] + 1
            stop_sequence[sel[-1]] = i+1+1 #to make up for matlab index
            stop_count = stop_count + 1
        i = i+1
    


    # breaks are the average of the stop sequences
    mybreaks = np.zeros((n,1))
    for i in range(stop_count):

        sel = stop_sequence == i # select the stop sequence
        sel=sel.flatten()    

        if i==0: # begin of stroke
            idxs = [idx for idx,b in enumerate(sel != 0) if b==True]
            mybreaks[i] = int(idxs[0]) # matlab find(sel,1,'first')
            
        elif i == stop_count-1: # end of stroke
            idxs = [idx for idx,b in enumerate(sel != 0) if b==True]
            mybreaks[i] = int(idxs[-1]) # matlab find(sel,1,'last')

        else: # all other positions            
            idxs = [idx for idx,b in enumerate(sel != 0) if b==True]
            mybreaks[i] = int(round(np.mean(idxs))) # find mean element
            
        mybreaks = mybreaks.astype(int)
                
        # set mean element to mean of sequence
        modified_input_stroke[mybreaks[i],:] = np.mean(modified_input_stroke[sel,:])
        
        # mark to keep
        stop_sequence[mybreaks[i]] = -1
        
    # Remove all other stop sequence elements, 
    # except for the marked mean
    idxs = [idx for idx,b in enumerate(stop_sequence > 0) if b==True]
    np.delete(modified_input_stroke,idxs,0)
    np.delete(stop_sequence,idxs,0)
    breaks = stop_sequence<0

    breaks[0]=True # MARK: THIS FIXES THINGS BUT WHY DO I NEED TO DO IT!!!!!!!!

    
    fbreaks = [idx for idx,b in enumerate(breaks != 0) if b==True]

    nb = len(fbreaks)
    ns = max(1,nb-1)
    
    substrokes = {}
    if nb==1: # if this stroke was just a single stop sequence
        assert(len(modified_input_stroke)==1)
    else:
        for s in range(ns):
            if s < ns-1:
                ss = modified_input_stroke[fbreaks[s]:fbreaks[s+1],:]
            else:
                ss = modified_input_stroke[fbreaks[s]:,:]
            substrokes[s] = ss


    new_start = substrokes[0][0,:]
    new_end = substrokes[len(substrokes)-1][-1,:]

    assert np.array_equal(new_start,modified_input_stroke[0,:])
    assert np.array_equal(new_end,modified_input_stroke[-1,:])
    
    return substrokes,modified_input_stroke,breaks

def unif_space(stroke,dist_int=1.0):

    '''
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_space_lerp.m
   
    Converts a stroke [x,y] so that it is uniformly sampled in space

    Inputs:
        - stroke [n x 2]
        - space_int: we want approximately this much distance (norm)
                     covered between successive points for spatial interpolation
        
    Outputs:
        - new_stroke [m x 2] interpolated stroke
    '''
    
    num_steps = len(stroke)

    # return if stroke is too short
    if len(stroke) == 1:
        return stroke

    # compute distance between each point
    dist = np.zeros((num_steps,1))
    to_remove = np.zeros((num_steps,1),dtype=bool) # array of false

    for i in range(2,num_steps):
        
        xy_1 = stroke[i]
        xy_2 = stroke[i-1]
        diff = xy_1 - xy_2
        dist[i] = np.linalg.norm(diff)
        to_remove[i] = dist[i] < 1e-4

    remove_indices = [i for i,b in enumerate(to_remove) if b==True]


    # remove points that are too close
    stroke = np.delete(stroke,remove_indices,axis=0)
    dist = np.delete(dist,remove_indices,axis=0)
    
    if len(stroke)==1:
        return stroke
    
    # cumulative distance
    cumdist = np.cumsum(dist)
    start_dist = cumdist[0]
    end_dist = cumdist[-1]
    nint = round(end_dist/dist_int)
    nint = int(max(nint,2))
    fx = interp1d(cumdist,stroke[:,0])
    fy = interp1d(cumdist,stroke[:,1])
    
    # new stroke
    query_points = np.linspace(start_dist,end_dist,nint,endpoint=True)
    new_stroke = np.zeros((len(query_points),2))
    new_stroke[:,0] = fx(query_points)
    new_stroke[:,1] = fy(query_points)
    
    return new_stroke



