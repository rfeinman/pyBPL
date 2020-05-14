import math
import numpy as np
import torch

from ...util import aeq
from ...data import unif_space
from ...splines import fit_bspline_to_traj, get_stk_from_bspline



def fit_smooth_stk(stroke, loss_threshold=0.3, max_nland=100):
    """
    Compute a "smooth" version of the original stroke by finding
    the minimal spline that meets a criterion of residual loss
    """
    ntraj = len(stroke)
    if ntraj == 1:
        return stroke

    stroke = unif_space(stroke)
    stroke = torch.tensor(stroke, dtype=torch.float)
    for nland in range(1, min(max_nland, ntraj+1)):
        spline, residuals = fit_bspline_to_traj(stroke, nland, include_resid=True)
        loss = torch.sum(residuals) / ntraj
        if loss.item() < loss_threshold:
            break
    stroke_ = get_stk_from_bspline(spline)
    stroke_ = stroke_.numpy()
    return stroke_


def split_by_junction(junct_pt, traj, radius):
    """
    Get portion of trajectory within the specific radius,
    and divide it in two based on the closest point to the junction
    """
    # compute distance of every pt from the junction pt
    d = np.linalg.norm(traj - junct_pt, axis=1)
    # get the relevant trajectory portion
    is_valid = d < radius
    # if we are re-tracing a previously visited junction, we
    # must be careful not to include any of the trajectory points
    # on this junction.
    last = np.where(is_valid)[0][-1]
    for i in reversed(range(last)):
        if not is_valid[i+1]:
            is_valid[i] = False
    # select valid set
    new_traj = traj[is_valid]
    d = d[is_valid]
    # divide into two halves
    bindx = np.argmin(d)
    first_half = new_traj[:bindx+1]
    second_half = new_traj[bindx+1:]

    return first_half, second_half


def compute_angle(seg_ext, seg_prev, ps):
    """
    Compute the angle between two vectors
    """
    n_ext = len(seg_ext)
    n_prev = len(seg_prev)
    if n_ext < 2 or n_prev < 2:
        return ps.faux_angle_too_short

    v_ext = seg_ext[-1] - seg_ext[0]
    v_prev = seg_prev[-1] - seg_prev[0]
    denom = np.linalg.norm(v_ext) * np.linalg.norm(v_prev)
    if aeq(denom, 0):
        denom = 1.
    val = np.dot(v_ext, v_prev) / denom
    val = min(val, 1)
    val = max(val, -1)
    thetaD = math.degrees(math.acos(val))
    return thetaD


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