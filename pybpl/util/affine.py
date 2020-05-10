import torch

from .stroke import com_char


__all__ = ['apply_warp']


def _warp_stk(stk, affine):
    """
    Helper; apply affine warp to a single stroke

    Parameters
    ----------
    stk : torch.Tensor
        (n,2) stroke trajectory
    affine : torch.Tensor
        (4,) affine parameter vector

    Returns
    -------
    stk : torch.Tensor
        (n,2) warped stroke trajectory

    """
    stk = stk * affine[:2] + affine[2:]
    return stk

def apply_warp(motor, A):
    """
    Apply affine warp to a character
    Reference: BPL/classes/MotorProgram.m (lines 231-245)

    Parameters
    ----------
    motor : list[torch.Tensor]
        a list of (m,n,2) or (n,2) tensors; collection of strokes (or stacked
        sub-strokes) that make up the character
    A : torch.Tensor
        (4,) affine warp

    Returns
    -------
    motor : list[torch.Tensor]
        warped strokes
        
    """
    cell_traj = torch.cat(motor) # (ns*m, n, 2) or (ns*n,2)
    com = com_char(cell_traj)
    B = torch.zeros(4)
    B[:2] = A[:2]
    B[2:] = A[2:] - (A[:2]-1)*com
    motor = [_warp_stk(stk, B) for stk in motor]

    return motor