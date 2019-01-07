function omniglot_extract_splines(input_type)
    switch input_type        
        case 'train'
            is_train = true;
        case 'test'
            is_train = false;
        otherwise
            error('invalid type');
    end

    if is_train
        load('data_background','drawings','images','names','timing');
    else
        load('data_evaluation','drawings','images','names','timing');
    end

    
    %% extract sub-strokes
    
    % Parameters
    ps = defaultps_preprocess;
    tint = ps.tint;

    % Augment with time as third dimension
    drawings_aug = drawings;
    nalpha = length(drawings);
    for a=1:nalpha
        nchar = length(drawings{a});
        for c=1:nchar
            nrep = length(drawings{a}{c});
            for r=1:nrep
                ns = length(drawings{a}{c}{r});
                for s=1:ns
                   drawings_aug{a}{c}{r}{s}(:,3) = timing{a}{c}{r}{s}; 
                end
            end
        end
    end

    % Convert to uniform time
    ufunc = @(stk) uniform_time_lerp(stk(:,1:2),stk(:,3),tint);
    udrawings = apply_to_nested(drawings_aug,ufunc);

    % Partition into sub-strokes
    pfunc = @(stk) partition_strokes(stk,ps.dthresh,ps.max_sequence);
    pdrawings = apply_to_nested(udrawings,pfunc);
    
    % Convert to uniform in space
    sfunc = @(stk) uniform_space_lerp(stk,ps.space_int);
    pdrawings = apply_to_nested(pdrawings,sfunc);
    
    
    %% extract splines and scales
    ps = defaultps_clustering();
    
    % remove strokes that are short in distance and time
    pdrawings = remove_short_stk(pdrawings,ps.minlen_ss,inf);
            % remove anything less than specific length

    % normalize the sub-strokes for clustering
    [pdrawings_norm,~,pdrawings_scales] = normalize_dataset(pdrawings,ps.newscale_ss);

    % fit b-spline to each of the sub-strokes
    bspline_substks = apply_to_nested(pdrawings_norm,@(x)fit_bspline_to_traj(x,ps.ncpt));

    % save the results
    if is_train
        save('data_background_splines','bspline_substks','pdrawings_norm','pdrawings_scales');
    else
        save('data_evaluation_splines','bspline_substks','pdrawings_norm','pdrawings_scales');
    end
end