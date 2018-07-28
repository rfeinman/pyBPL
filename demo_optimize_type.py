"""
Sample a character type and then optimize its parameters to maximize the
likelihood of the type under the prior
"""
from __future__ import print_function, division
import argparse
import matplotlib.pyplot as plt
import torch

from pybpl.library.library import Library
from pybpl.character.ctd import CharacterTypeDist
from pybpl.character.character import Character

dtype = torch.float

# use projected gradient ascent for constrained optimization
projected_grad_ascent = True
# learning rate
lr = 1e-3
# tol. for constrained optimization
eps = 1e-4


def get_variables(S, R):
    """
    Indicate variables for optimization (requires_grad_)

    :param S: [list of Stroke]
    :param R: [list of Relation]
    :return:
        parameters: [list] list of optimizable parameters
        lbs: [list] list of lower bounds (each elem. is a tensor same size
                as param; empty list indicates no lb)
        ubs: [list] list of upper bounds (each elem. is a tensor; empty
                list indicates no ub)
    """
    parameters = []
    lbs = []
    ubs = []
    for s, r in zip(S, R):
        # shape
        s.shapes_type.requires_grad_()
        parameters.append(s.shapes_type)
        lbs.append([])
        ubs.append([])

        # scale
        s.invscales_type.requires_grad_()
        parameters.append(s.invscales_type)
        lbs.append(torch.full(s.invscales_type.shape, eps))
        ubs.append([])

    return parameters, lbs, ubs


def obj_fun(S, R, ctd):
    """
    Evaluate the log-likelihood of a character type under the prior

    :param S: [list of Stroke]
    :param R: [list of Relation]
    :return:
        ll: [tensor] log-likelihood under the prior. Scalar
    """
    # start the accumulator
    ll = 0.

    # loop through the strokes
    for s, r in zip(S, R):
        # log-prob of the control points for each sub-stroke in this stroke
        ll_cpts = ctd.score_shapes_type(s.ids, s.shapes_type)
        # log-prob of the scales for each sub-stroke in this stroke
        ll_scales = ctd.score_invscales_type(s.ids, s.invscales_type)
        # sum over sub-strokes and add to accumulator
        ll = ll + torch.sum(ll_cpts) + torch.sum(ll_scales)

    return ll


def main():
    # load the library
    lib = Library(lib_dir='./lib_data')
    # generate a character type
    ctd = CharacterTypeDist(lib)
    S, R = ctd.sample_type(k=args.ns)
    char = Character(S, R, lib)
    ex1 = char.sample_token().image.numpy()
    ex2 = char.sample_token().image.numpy()
    print('num strokes: %i' % len(S))
    # get optimizable variables & their bounds
    parameters, lbs, ubs = get_variables(S, R)

    # optimize the character type
    score_list = []
    for idx in range(1000):
        if idx % 100 == 0:
            print('iteration #%i' % idx)
        score = obj_fun(S, R, ctd)
        score.backward(retain_graph=True)
        score_list.append(score)
        with torch.no_grad():
            for ip, param in enumerate(parameters):
                # manual grad. ascent
                param.add_(lr*param.grad)
                if projected_grad_ascent:
                    lb = lbs[ip]
                    ub = ubs[ip]
                    if len(lb)>0:
                        torch.max(param, lb, out=param)
                    if len(ub)>0:
                        torch.min(param, ub, out=param)

                param.grad.zero_()

    ex3 = char.sample_token().image.numpy()
    ex4 = char.sample_token().image.numpy()

    plt.figure()
    plt.plot(score_list)
    plt.ylabel('log-likelihood')
    plt.xlabel('iteration')

    fig, axes = plt.subplots(2,2,figsize=(10,8))
    axes[0,0].imshow(ex1, cmap='Greys', vmin=0, vmax=1)
    axes[0,0].set_title('pre-train ex1')
    axes[0,1].imshow(ex2, cmap='Greys', vmin=0, vmax=1)
    axes[0,1].set_title('pre-train ex2')
    axes[1,0].imshow(ex3, cmap='Greys', vmin=0, vmax=1)
    axes[1,0].set_title('post-train ex1')
    axes[1,1].imshow(ex4, cmap='Greys', vmin=0, vmax=1)
    axes[1,1].set_title('post-train ex2')

    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ns', help="The number of strokes",
        required=False, type=int
    )
    args = parser.parse_args()
    main()