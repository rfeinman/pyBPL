import numpy as np



def fit_smooth_stk(stk, image, ps):
    raise NotImplementedError

def split_by_junction(junct_pt, traj, radius):
    """
    Get portion of trajectory within the specific radius,
    and divide it in two based on the closest point to the junction
    """
    raise NotImplementedError

def compute_angle(seg_ext, seg_prev, ps):
    """
    Compute the angle between two vectors
    """
    raise NotImplementedError

def stroke_from_nodes(graph, list_ni):
    """
    Compute stroke trajectory from a graph and a list of nodes
    """
    num_nodes = len(list_ni)
    if num_nodes == 1:
        # edge case: single-point stroke
        pt = graph.nodes[list_ni[0]]['o']
        return pt[None]
    stroke = []
    for i in range(1, num_nodes):
        curr_ni = list_ni[i-1]
        curr_pt = graph.nodes[curr_ni]['o']
        next_ni = list_ni[i]
        traj = graph.edges[curr_ni, next_ni]['pts']
        d_start = np.linalg.norm(curr_pt - traj[0])
        d_end = np.linalg.norm(curr_pt - traj[-1])
        if d_end < d_start: # flip the traj if needed
            traj = np.flip(traj, 0)
        traj = np.insert(traj, 0, curr_pt, axis=0)
        stroke.append(traj)
    stroke = np.concatenate(stroke)
    return stroke