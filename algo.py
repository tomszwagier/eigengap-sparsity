import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
from scipy.optimize import isotonic_regression
from scipy.stats import ortho_group
from time import time

from psa import candidate_models, model_selection


def eigengap_cst(p):
    return np.array([1] + [0] * (p - 2) + [-1]) + np.array([(p - 2 * s + 1) for s in range(1, p + 1)])


def penalized_normal_loglikelihood(lbda, V, X):
    p, n = X.shape
    S = (1 / n) * X @ X.T
    nu = (np.log(n) / n) * eigengap_cst(p)
    return np.sum(np.log(lbda)) + np.trace(V @ np.diag(1. / lbda) @ V.T @ S) + np.dot(nu, lbda)


def project_orthogonal(V):
    u, _, vt = np.linalg.svd(V)
    return u @ vt


def project_cone(lbda):
    res = isotonic_regression(lbda, weights=None, increasing=False)  # N.B: if at some point I need the flag type, it can be obtained by `np.diff(res.blocks)`.
    return np.clip(res.x, 1e-10, 1e10)


def pcd(loss_func, X, lr=0.01, max_iter=1000, tol=1e-6):
    steps = []
    count = 0
    V, lbda = ortho_group.rvs(dim=X.shape[0]), np.random.normal(loc=1, scale=1.0, size=X.shape[0])  # TODO: CAUTION init?
    prev_loss = np.inf
    while count < max_iter:
        V = V - lr * grad(lambda V_: loss_func(lbda, V_, X))(V)
        V = project_orthogonal(V)
        lbda = lbda - lr * grad(lambda lbda_: loss_func(lbda_, V, X))(lbda)
        lbda = project_cone(lbda)
        new_loss = loss_func(lbda, V, X)
        steps.append(new_loss)
        if (count > 0) and (abs((prev_loss - new_loss) / prev_loss) < tol):  # replace with criterion on the gradient?
            break
        count += 1
        prev_loss = new_loss
    return V, lbda, steps


if __name__ == "__main__":  # TODO: should not we remove sample mean even if we assume it is 0?
    np.random.seed(42)

    n, p = 40, 20  # 20, 10 / 40, 20 / 200, 100 / ...
    X = np.random.multivariate_normal(mean=np.zeros(p,), cov=np.eye(p), size=n).T

    # Sample covariance matrix
    start = time()
    S = (1 / n) * X @ X.T
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = eigval[::-1], eigvec[::-1]
    print("Time sample cov. =", time() - start)
    print("Pen. lik. sample cov. =", penalized_normal_loglikelihood(eigval, eigvec, X))
    print("Eigenvalues sample cov.", eigval)

    # Principal subspace analysis
    start = time()
    print("begin")
    models = candidate_models(p)
    print("done")
    model_best, eigval_psa, eigvec_psa = model_selection(X, models)
    print("Time PSA =", time() - start)
    print("Pen. lik. PSA =", penalized_normal_loglikelihood(eigval_psa, eigvec_psa, X))
    print("Eigenvalues PSA", eigval_psa)

    # Eigengap sparsity
    start = time()
    V, lbda, steps = pcd(penalized_normal_loglikelihood, X, lr=0.01, max_iter=1000, tol=1e-6)
    print("Time ESCP =", time() - start)
    print("Pen. lik. ESCP =", penalized_normal_loglikelihood(lbda, V, X))
    print("Eigenvalues ESCP", lbda)
    plt.figure()
    plt.plot(steps)
    plt.show(block=True)


