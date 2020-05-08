% Mostly bottom-up method for generating a set
% See https://github.com/brendenlake/BPL/blob/master/bottomup/generate_random_parses.m
function S_walks = generate_random_parses_RF(I,seed)
    if exist('seed', 'var')
        rng(seed);
    end

    ps = defaultps;
    load(ps.libname,'lib');

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
    ps = defaultps_bottomup;
    RW = RandomWalker(G);
    PP = ProcessParsesRF(I,lib,false);
    
    % Add deterministic minimum angle walks
    for i=1:ps.nwalk_det
        PP.add(RW.det_walk);
    end
    
    % Sample random walks until we reach capacity.
    walk_count = PP.nwalks;
    while (PP.nl < ps.max_nstroke) && (walk_count < ps.max_nwalk)
        list_walks = RW.sample(1);
        PP.add(list_walks{1});
        walk_count = walk_count + 1;
    end

    PP.freeze;
    S_walks = PP.get_S;
    
end