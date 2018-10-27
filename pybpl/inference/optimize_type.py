from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import torch



def get_optimizable_variables(c, eps):
    """
    Indicate variables for optimization (requires_grad_)

    Parameters
    ----------
    c : Character
        character type to optimize
    eps : float
        tol. for constrained optimization

    Returns
    -------
    parameters : list
        list of optimizable parameters
    lbs : list
        list of lower bounds (each elem. is a tensor same size as param;
        empty list indicates no lb)
    ubs : list
        list of upper bounds (each elem. is a tensor; empty list indicates
        no ub)
    """
    parameters = []
    lbs = []
    ubs = []
    for p in c.P:
        # shape
        p.shapes.requires_grad_()
        parameters.append(p.shapes)
        lbs.append([])
        ubs.append([])

        # scale
        p.invscales.requires_grad_()
        parameters.append(p.invscales)
        lbs.append(torch.full(p.invscales.shape, eps))
        ubs.append([])

    return parameters, lbs, ubs


def optimize_type(
        c, type_dist, lr, nb_iter, eps, projected_grad_ascent,
        show_examples=True
):
    """
    Take a character type and optimize its parameters to maximize the
    likelihood under the prior, using gradient descent

    Parameters
    ----------
    c : Character
        TODO
    type_dist : CharacterTypeDist
        TODO
    lr : float
        TODO
    nb_iter : int
        TODO
    eps : float
        TODO
    projected_grad_ascent : bool
        TODO
    show_examples : bool
        TODO

    Returns
    -------
    score_list : list of float
        TODO

    """
    # get optimizable variables & their bounds
    parameters, lbs, ubs = get_optimizable_variables(c, eps)
    # optimize the character type
    score_list = []
    if show_examples:
        n_plots = nb_iter//100
        fig, axes = plt.subplots(nrows=n_plots, ncols=4, figsize=(4, n_plots))
    for idx in range(nb_iter):
        if idx % 100 == 0 and show_examples:
            print('iteration #%i' % idx)
            for i in range(4):
                _, ex = c.sample_token()
                axes[idx//100, i].imshow(ex.numpy(), cmap='Greys', vmin=0, vmax=1)
                axes[idx//100, i].tick_params(
                    which='both',
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False
                )
            axes[idx//100, 0].set_ylabel('%i' % idx)
        score = type_dist.score_type(c)
        score.backward(retain_graph=True)
        score_list.append(score)
        with torch.no_grad():
            for ip, param in enumerate(parameters):
                # manual grad. ascent
                param.add_(lr * param.grad)
                if projected_grad_ascent:
                    lb = lbs[ip]
                    ub = ubs[ip]
                    if len(lb) > 0:
                        torch.max(param, lb, out=param)
                    if len(ub) > 0:
                        torch.min(param, ub, out=param)

                param.grad.zero_()

    return score_list