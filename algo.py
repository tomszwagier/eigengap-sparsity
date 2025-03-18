import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from scipy.optimize import isotonic_regression
from scipy.stats import ortho_group
from sklearn.covariance import ledoit_wolf
from time import time

from psa import candidate_models, model_selection, sep_to_type, kappa


def eigengap_cst(p):
    return np.array([1] + [0] * (p - 2) + [-1]) + np.array([(p - 2 * s + 1) for s in range(1, p + 1)])


def penalized_normal_loglikelihood(lbda, V, X):
    p, n = X.shape
    S = (1 / n) * X @ X.T
    nu = (np.log(n) / n) * eigengap_cst(p)
    return np.sum(np.log(lbda)) + np.trace(V @ np.diag(1. / lbda) @ V.T @ S) + np.dot(nu, lbda)


def penalized_normal_loglikelihood_l0(lbda, V, X):
    p, n = X.shape
    S = (1 / n) * X @ X.T
    return np.sum(np.log(lbda)) + np.trace(V @ np.diag(1. / lbda) @ V.T @ S) + (np.log(n) / n) * kappa(sep_to_type(np.diff(lbda)))


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


if __name__ == "__main__":
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
    print("Pen. lik. l0 sample cov. =", penalized_normal_loglikelihood_l0(eigval, eigvec, X))
    print("Frob. loss sample cov. =", (1 / p) * np.linalg.norm((np.eye(p) - S), ord='fro')**2)
    print("Eigenvalues sample cov.", eigval)
    print("Num params sample cov.", kappa(sep_to_type(np.diff(eigval))))

    # Principal subspace analysis
    start = time()
    models = candidate_models(p)
    model_best, eigval_psa, eigvec_psa = model_selection(X, models)
    print("Time PSA =", time() - start)
    print("Pen. lik. PSA =", penalized_normal_loglikelihood(eigval_psa, eigvec_psa, X))
    print("Pen. lik. l0 PSA =", penalized_normal_loglikelihood_l0(eigval_psa, eigvec_psa, X))
    print("Frob. loss PSA =", (1 / p) * np.linalg.norm((np.eye(p) - eigvec_psa @ np.diag(eigval_psa) @ eigvec_psa.T), ord='fro')**2)
    print("Eigenvalues PSA", eigval_psa)
    print("Num params PSA", kappa(sep_to_type(np.diff(eigval_psa))))

    # Eigengap sparsity
    start = time()
    V, lbda, steps = pcd(penalized_normal_loglikelihood, X, lr=0.01, max_iter=1000, tol=1e-6)
    print("Time ESCP =", time() - start)
    print("Pen. lik. ESCP =", penalized_normal_loglikelihood(lbda, V, X))
    print("Pen. lik. l0 ESCP =", penalized_normal_loglikelihood_l0(lbda, V, X))
    print("Frob. loss ESCP =", (1 / p) * np.linalg.norm((np.eye(p) - V @ np.diag(lbda) @ V.T), ord='fro')**2)
    print("Eigenvalues ESCP", lbda)
    print("Num params ESCP", kappa(sep_to_type(np.diff(lbda))))
    plt.figure()
    plt.plot(steps)
    plt.show(block=True)

    # Ledoit-Wolf
    start = time()
    shrunk_cov, shrinkage = ledoit_wolf(X.T, assume_centered=True)
    eigval_LW, eigvec_LW = np.linalg.eigh(shrunk_cov)
    eigval_LW, eigvec_LW = eigval_LW[::-1], eigvec_LW[::-1]
    print("Time LW =", time() - start)
    print("Pen. lik. LW =", penalized_normal_loglikelihood(eigval_LW, eigvec_LW, X))
    print("Pen. lik. l0 LW =", penalized_normal_loglikelihood_l0(eigval_LW, eigvec_LW, X))
    print("Frob. loss LW =", (1 / p) * np.linalg.norm((np.eye(p) - eigvec_LW @ np.diag(eigval_LW) @ eigvec_LW.T), ord='fro')**2)
    print("Eigenvalues LW", eigval_LW)
    print("Num params LW", kappa(sep_to_type(np.diff(eigval_LW))))
