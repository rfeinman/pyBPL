% Mostly bottom-up method for generating a set
% See https://github.com/brendenlake/BPL/blob/master/bottomup/generate_random_parses.m
function S_walks = generate_random_parses_RF(I,seed,max_ntrials,max_nwalk,max_nstroke,nwalk_det)
    % apply random seed
    if exist('seed', 'var')
        rng(seed);
    end

    % load library
    ps = defaultps; load(ps.libname,'lib');

    % load default parameters
    ps = defaultps_bottomup;
    if ~exist('max_ntrials', 'var')
        max_ntrials = ps.max_nwalk;
    end
    if ~exist('max_nwalk', 'var')
        max_nwalk = ps.max_nwalk;
    end
    if ~exist('max_nstroke', 'var')
        max_nstroke = ps.max_nstroke;
    end
    if ~exist('nwalk_det', 'var')
        nwalk_det = ps.nwalk_det;
    end

    % Check that image is in the right format    
    assert(UtilImage.check_black_is_true(I));
    
    % If we have a blank image
    if sum(sum(I))==0
       bestMP = [];
       return
    end
    
    % Get character skeleton from the fast bottom-up method
    G = extract_skeleton(I);
    
    % Create a set of random parses through random walks
    RW = RandomWalker(G);
    PP = ProcessParsesRF(I,lib,false);
    
    % Add deterministic minimum angle walks
    for i=1:nwalk_det
        PP.add(RW.det_walk);
    end
    
    % Sample random walks until we reach capacity.
    ntrials = PP.nwalks;
    while (PP.nl < max_nstroke) && (PP.nwalks < max_nwalk) && (ntrials < max_ntrials)
        list_walks = RW.sample(1);
        PP.add(list_walks{1});
        ntrials = ntrials + 1;
    end

    PP.freeze;
    S_walks = PP.get_S;
    
end