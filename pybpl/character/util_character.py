from __future__ import division, print_function
import torch

def clear_shape_type(M):
    raise NotImplementedError

def clear_invscales_type(M):
    raise NotImplementedError

def set_affine_to_image(M, I):
    raise NotImplementedError

def set_affine(M, target_com, target_range, missing):
    raise NotImplementedError

def set_affine_motor(M, target_com, target_range):
    raise NotImplementedError

def apply_each_substroke(nested, fnc):
    raise NotImplementedError
    nested = None

    return nested

def flatten_substrokes(motor_list):
    """
    Flatten the sub-stroke fine motor sequence.

    Takes a list of (nsub,neval,2)-shape tensors and concatenates them
    along the 0-th axis. This is more complicated then a simple reshape, since
    nsub is different for each entry (otherwise motor_list could be a tensor and
    we could simply use Tensor.view).

    :param motor_list: [list of tensors] ns-length list of (nsub,neval,2)
                        tensors; the fine motor sequence of each stroke
    :return:
        mlist_flat: [(nsub_total,neval,2) tensor] flattened motor sequence for
                        the character
    """
    for motor in motor_list:
        assert isinstance(motor, torch.Tensor)
    # store number of evaluations
    neval = motor_list[0].shape[1]
    # make sure all strokes have same number of evaluations
    for motor in motor_list:
        assert motor.shape[1] == neval
    # count total number of sub-strokes in the character
    nsub_total = sum([motor.shape[0] for motor in motor_list])

    # build mlist_flat
    mlist_flat = torch.zeros(nsub_total, neval, 2)
    ss_id = 0
    for motor in motor_list:
        nsub = motor.shape[0]
        for bid in range(nsub):
            mlist_flat[ss_id] = motor[bid]
            ss_id += 1

    return mlist_flat

def flip_stroke(S):
    raise NotImplementedError

def merge_strokes(S1, S2):
    raise NotImplementedError
    Z = None

    return Z

def split_stroke(Z, bid_start_of_second):
    raise NotImplementedError
    Z1 = None
    Z2 = None

    return Z1, Z2

def all_merge_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def all_split_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def valid_merge(M, sid):
    raise NotImplementedError
    val = None

    return val