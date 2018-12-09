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


def partition_stroke(stroke,dthresh=1,max_seq=np.infty):
    '''
    NOTE: NOT YET FINISHED
    '''
    return {},stroke,True # PSEUDO OUTPUT FOR NOW
    
    
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
    n = len(stroke)
    dxdt = get_deriv(stroke,1:n)

    # Special case
    if n==1:
        substrokes = {0:stroke}
        return substrokes,stroke,True
    
    # compute norm of derivs
    norm_dxdt = np.zeros((n,1))
    for i in range(n):
        norm_dxdt[i] = np.linalg.norm(dxdt[i])

    # compute candidate stop poitns
    stop_pt = norm_dxdt < dthresh
    for i in range(2,n):
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
   
    '''

    stop_sequence = zeros(n,1);
    stop_count = 1;
    for i=1:n        
        if stop_pt(i) % current point is a stop, it's the same stop
            stop_sequence(i) = stop_count;
        elseif stop_pt(i-1) && stop_pt(i+1) % points surround it are a stop... its the same stop
            stop_sequence(i) = stop_count;
        elseif stop_pt(i-1)
            stop_count = stop_count + 1; % we just finishsed a stop
        end
    end
    
    % Special case where the entire stroke is a stop sequence
    if stop_count == 1
        stop_sequence = zeros(n,1);
        stop_sequence(1) = 1;
        stop_sequence(end) = 2;
        stop_count = 2;
    end    
    
    % Make sure the stop sequences aren't too long. If they are,
    % we place a sub-stroke break at the beginning and end.
    i = 1;
    while i <= stop_count
        sel = find(stop_sequence==i);
        nsel = length(sel);
        if nsel>max_sequence
            stop_sequence(sel(2:end)) = 0;
            stop_sequence(stop_sequence>i) = stop_sequence(stop_sequence>i) + 1;
            stop_sequence(sel(end)) = i+1;
            stop_count = stop_count + 1;
        end
        i = i + 1;
    end    
    
    % Breaks are the average of the stop sequences
    mybreaks = zeros(n,1);
    for i=1:stop_count
        sel = stop_sequence==i; % select the stop sequence
        
        if i==1 % beginning of stroke
            mybreaks(i) = find(sel,1,'first');
        elseif i==stop_count % end of stroke
            mybreaks(i) = find(sel,1,'last');
        else % all other positions
            mybreaks(i) = round(mean(find(sel))); % find the mean element
        end
            
        % Set the mean element to the mean of the sequence
        unif_stk(mybreaks(i),:) = mean(unif_stk(sel,:),1);
        
        % mark to keep
        stop_sequence(mybreaks(i)) = -1;
    end
    
    % Remove all other stop sequence elements, 
    % except for the marked mean
    unif_stk(stop_sequence>0,:) = [];
    stop_sequence(stop_sequence>0) = [];
    breaks = stop_sequence<0;
    
    % Convert to cell array
    fbreaks = find(breaks);
    nb = length(fbreaks);
    ns = max(1,nb-1);
    substrokes = cell(ns,1);
    if nb==1 % if this stroke was just a single stop sequence
        assert(size(unif_stk,1)==1);
        substrokes{1} = unif_stk;
    else
        for s=1:ns
            substrokes{s} = unif_stk(fbreaks(s):fbreaks(s+1),:);
        end
    end
    
    new_start = substrokes{1}(1,:);
    new_end = substrokes{end}(end,:);
    assert( aeq(new_start,unif_stk(1,:)) );
    assert( aeq(new_end,unif_stk(end,:)) );
end

% Compute dx/dt partial derivatives
% for each column (variable) of X
% 
% Input
%  X: [T x dim]
%  t: [T x 1] time points
% 
% Output
%  dxdt: [T-1 x 1] derivatives
function dxdt = get_deriv(X,t)

    [T,dim] = size(X);
    assert(isvector(t));
    assert(numel(t)==T);
    dxdt = zeros(size(X));
    
    for i=2:T
        prev = X(i-1,:);
        next = X(i,:);
        dt = t(i)-t(i-1);
        dxdt(i,:) = (next-prev)./dt;
    end
    
end

    '''

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



