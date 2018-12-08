import numpy as np

from scipy.interpolate import interp1d


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

        first_strokes = np.vstack(first_strokes)
        return first_strokes




# inputs: 
# stroke assumes uniform time
# dthresh: if this much distance is not covered
# at each time point, then it's a pause
# max_sequence: maximum length of a stop sequence
#               before it is called it's own stroke

# outputs:
    # substrokes [ns x 1] dict of substrokes, each of which
    # follows stroke indexing convention
    # unif_stk: stroke with pause sequences shortened to a single point
    # breaks: where pauses occured
def parition_strokes(stroke,dthresh=1,max_sequence=np.infty):
    return stroke
    '''
    not implemented, translate from https://github.com/brendenlake/BPL/blob/master/data/partition_strokes.m
    '''


# Convert a stroke [x,y,t] so that it is uniformly sampled in space
# space_int:  we want approximately this much distance (norm)
#             covered between successive points for spatial interpolation
def unif_space(stroke,dist_int=1.0):
    return stroke

    '''
    # Not done: translate from https://github.com/brendenlake/BPL/blob/master/data/uniform_space_lerp.m

    num_steps = len(stroke)

    # return if stroke is too short
    if len(stroke) == 1:
        return stroke

    
    # compute distance between each point
    dist = np.zeros((num_steps,1))
    to_remove = np.zeros((num_steps,1),dtype=bool) # array of false

    for i in range(2,num_steps):
        xy_1 = stroke[i,:2]
        xy_2 = stroke[i-1,:2]
        dist[i] = np.square(xy_1 - xy_2).sum()
        to_remove[i] = dist[i] < 1e-4
    

    remove_indices = [i for i,b in enumerate(to_remove) if b==True]


    # remove points that are too close
    stroke = np.delete(stroke,remove_indices,axis=0)
    dist = np.delete(dist,remove_indices,axis=0)
    
    if len(stroke)==1:
        return stroke
    
    new_stroke = np.zeros((len(stroke),3))

    # cumulative distance
    cumdist = np.cumsum(dist)
    start_dist = cumdist[0]
    end_dist = cumdist[-1]
    x = cumdist
    
    print("cumdist shape",x.shape)
    print("stroke shape",stroke.shape)

    nint = round(end_dist/dist_int)
    nint = max(nint,2)
    query_points = np.linspace(start_dist,end_dist,nint)
    
    new_stroke = np.zeros((len(stroke),3))
    f = interp1d(x,stroke[:,:2])
    new_stroke[:,2] = f(query_points)
    new_stroke[:,2] = stroke[:,2]

    return new_stroke
    '''

# Convert a stroke [x,y,t] so that it is uniformly sampled in time
# time_int: interpolate stroke so that we have datapoints every
#           interval of this many milliseconds
def unif_time(stroke,time_int=50.0):

    '''
    Done, translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_time_lerp.m
    '''
    
    times = stroke[:,2]
    min_time = min(times)
    max_time = max(times)
    
    # range excludes endpoint
    unif_times = list(np.arange(min_time,max_time,time_int))
    unif_times.append(max_time)
    unif_times = np.array(unif_times)    
    unif_steps = len(unif_times)
    unif_stroke = np.zeros((unif_steps,3))
    
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
            unif_stroke[t][:2] = np.mean(matches,axis=0)[:2]
            unif_stroke[t][2] = new_time
          
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

            unif_stroke[t][:2] = (1-frac) * prev_xy + frac * post_xy
            unif_stroke[t][2] = new_time
    return unif_stroke
