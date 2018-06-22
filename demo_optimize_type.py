from __future__ import print_function, division
import matplotlib.pyplot as plt
import torch

from pybpl.classes import Library, CPD
from pybpl.forward_model import generate_character

dtype = torch.float

# use projected gradient ascent for constrained optimization
projected_grad_ascent = True
# learning rate
lr = 0.01
# tol. for constrained optimization
eps = 0.0001

def get_variables_MP(mp):
    """
    Indicate variables for optimization (requires_grad_)

    :param mp: [MotorProgram] the character type
    :return:
        parameters: TODO
        lbs: [list] list of lower bounds (each elem. is a tensor same size
                as param; empty list indicates no lb)
        ubs: [list ] list of upper bounds (each elem. is a tensor; empty
                list indicates no ub)
    """
    parameters = []
    lbs = []
    ubs = []
    for sid in range(mp.ns):
        # shape
        mp.S[sid].shapes_type.requires_grad_()
        parameters.append(mp.S[sid].shapes_type)
        lbs.append([])
        ubs.append([])

        # scale
        mp.S[sid].invscales_type.requires_grad_()
        parameters.append(mp.S[sid].invscales_type)
        lbs.append(torch.full(mp.S[sid].invscales_type.shape, eps))
        ubs.append([])

    return parameters, lbs, ubs

def obj_fun(mp, lib):
    """
    Evaluate the log-likelihood of a character type under the prior

    :param mp: [MotorProgram] the character type
    :return:
        ll: [tensor] log-likelihood under the prior. Scalar
    """
    # NOTE: don't declare requires_grad
    ll = torch.tensor(0., dtype=dtype)

    # loop through the strokes
    # NOTE: must use inline addition
    for sid in range(mp.ns):
        stroke = mp.S[sid]
        # log-prob of the control points for each sub-stroke in the sequence
        ll_cpts = CPD.score_shape_type(
            lib, stroke.shapes_type, stroke.ids
        )
        # log-prob of the scales for this sub-stroke sequence
        ll_scales = CPD.score_invscale_type(
            lib, stroke.invscales_type, stroke.ids
        )
        # sum over sub-strokes and add to accumulator
        ll.add_(torch.sum(ll_cpts))
        ll.add_(torch.sum(ll_scales))

    return ll

def main():
    # load the library
    lib = Library(lib_dir='./library')
    # generate a motor program (a character type)
    mp = generate_character(lib)
    # get optimizable variables & their bounds
    parameters, lbs, ubs = get_variables_MP(mp)

    score_list = []
    for idx in range(1000):
        score = obj_fun(mp)
        score.backward()
        score_list.append(score)
        with torch.no_grad():
            for ip, param in enumerate(parameters):
                # manual grad. ascent
                param.add_(lr*param.grad)
                if projected_grad_ascent:
                    lb = lbs[ip]
                    ub = ubs[ip]
                    if len(lb)>0:
                        param = torch.max(param, lb)
                    if len(ub)>0:
                        param = torch.min(param, ub)

                param.grad.zero_()

    plt.plot(score_list)
    plt.ylabel('log-likelihood')
    plt.xlabel('test')
    plt.show()

if __name__ == "__main__":
    main()