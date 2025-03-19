import itertools
import numpy as np


def sep_to_type(seps):
    model = [1]
    for sep in seps:
        if sep:
            model.append(1)
        else:
            model[-1] += 1
    return model


def candidate_models(p):
    """ Generate the whole family of Principal Subspace Analysis (PSA) models.
    """
    models = []
    for seps in list(itertools.product(*((False, True),)*(p-1))):
        model = sep_to_type(seps)
        models.append(model)
    return models


def kappa(model):
    """ Compute the number of free parameters of a PSA model.
    """
    p = np.sum(model)
    kappa_eigvals = len(model)
    kappa_eigenspaces = int(p * (p - 1) / 2 - np.sum(np.array(model) * (np.array(model) - 1) / 2))
    return kappa_eigvals + kappa_eigenspaces


def ll(model, eigval, n):
    """ Compute the maximum log-likelihood of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    p = np.sum(model)
    q_list = (0,) + tuple(np.cumsum(model))
    eigval_mle = np.concatenate([[np.mean(eigval[qk:qk_])] * gamma_k for (qk, qk_, gamma_k) in zip(q_list[:-1], q_list[1:], model)])
    return - (n / 2) * (p * np.log(2 * np.pi) + np.sum(np.log(eigval_mle)) + p), eigval_mle


def bic(model, eigval, n):
    """ Compute the Bayesian Information Criterion (BIC) of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    ll_, eigval_mle = ll(model, eigval, n)
    return kappa(model) * np.log(n) - 2 * ll_, eigval_mle


def model_selection(X, models):
    """ Perform model selection by minimizing the BIC among a family of candidate models.
    """
    p, n = X.shape
    S = (1 / n) * X @ X.T
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
    model_best, eigval_best, crit_best = None, None, np.inf
    for model in models:
        crit_model, eigval_mle = bic(model, eigval, n)
        if crit_model < crit_best:
            model_best, eigval_best, crit_best = model, eigval_mle, crit_model
    return model_best, eigval_best, eigvec


if __name__ == "__main__":
    print(candidate_models(5))
    print([np.sum(model) for model in candidate_models(5)])
