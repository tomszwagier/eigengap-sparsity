import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import isotonic_regression
from sklearn.covariance import ledoit_wolf
from time import time

from psa import candidate_models, model_selection, sep_to_type, kappa


def coeff(p, n):
    return (np.log(n) / n) * (np.array([1] + [0] * (p - 2) + [-1]) + np.array([(p - 2 * s + 1) for s in range(1, p + 1)]))


def penalized_normal_loglikelihood_l1(eigval, eigval_scm):
    p, n = X.shape
    return np.sum(np.log(eigval) + (eigval_scm / eigval) + coeff(p, n) * eigval)


def penalized_normal_loglikelihood_l0(eigval, eigval_scm):
    p, n = X.shape
    return np.sum(np.log(eigval) + (eigval_scm / eigval)) + (np.log(n) / n) * kappa(sep_to_type(np.diff(eigval)))


def project_cone(eigval):
    res = isotonic_regression(eigval, weights=None, increasing=False)
    return np.clip(res.x, 1e-10, 1e10)


def pgd(X, lr=0.01, max_iter=1000, tol=1e-6):
    steps = []
    count = 0
    S = (1 / n) * X @ X.T
    eigval_scm, _ = np.linalg.eigh(S)
    eigval_scm = eigval_scm[::-1]
    eigval = np.copy(eigval_scm)
    prev_loss = np.inf
    while count < max_iter:
        eigval = eigval - lr * ((1 / eigval) - (eigval_scm / eigval**2) + coeff(p, n))
        eigval = project_cone(eigval)
        new_loss = penalized_normal_loglikelihood_l1(eigval, eigval_scm)
        steps.append(new_loss)
        if (count > 0) and (abs((prev_loss - new_loss) / prev_loss) < tol):
            break
        count += 1
        prev_loss = new_loss
        # print(sep_to_type(np.diff(eigval)))
    return eigval, steps


if __name__ == "__main__":
    np.random.seed(42)

    # n, p = 40, 20  # 20, 10 / 40, 20 / 200, 100 / ...
    # X = np.random.multivariate_normal(mean=np.zeros(p,), cov=np.eye(p), size=n).T
    n, p = 600, 200
    X = np.random.multivariate_normal(mean=np.zeros(p,), cov=np.diag([10] * int(.4 * p) + [3] * int(.4 * p) + [1] * int(.2 * p)), size=n).T

    # Sample covariance
    print("SAMPLE COVARIANCE")
    start = time()
    S = (1 / n) * X @ X.T
    eigval_scm, eigvec_scm = np.linalg.eigh(S)
    eigval_scm, eigvec_scm = eigval_scm[::-1], eigvec_scm[:, ::-1]
    print("Time = ", time() - start)
    print("Objective = ", penalized_normal_loglikelihood_l0(eigval_scm, eigval_scm))
    print("Objective L1 = ", penalized_normal_loglikelihood_l1(eigval_scm, eigval_scm))
    print("Frob. loss = ", (1 / p) * np.linalg.norm((np.eye(p) - S), ord='fro')**2)
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_scm))))
    print("Eigenvalues = ", eigval_scm)

    # Ledoit-Wolf
    print("LEDOIT-WOLF")
    start = time()
    shrunk_cov, shrinkage = ledoit_wolf(X.T, assume_centered=True)
    eigval_lw, _ = np.linalg.eigh(shrunk_cov)
    eigval_lw = eigval_lw[::-1]
    print("Time = ", time() - start)
    print("Objective = ", penalized_normal_loglikelihood_l0(eigval_lw, eigval_scm))
    print("Objective L1 = ", penalized_normal_loglikelihood_l1(eigval_lw, eigval_scm))
    print("Frob. loss = ", (1 / p) * np.linalg.norm((np.eye(p) - shrunk_cov), ord='fro')**2)
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_lw))))
    print("Eigenvalues = ", eigval_lw)

    # # Principal subspace analysis
    # print("PRINCIPAL SUBSPACE ANALYSIS")
    # start = time()
    # models = candidate_models(p)
    # model_best, eigval_psa, eigvec_psa = model_selection(X, models)
    # print("Time = ", time() - start)
    # print("Objective = ", penalized_normal_loglikelihood_l0(eigval_psa, eigval_scm))
    # print("Objective L1 = ", penalized_normal_loglikelihood_l1(eigval_psa, eigval_scm))
    # print("Frob. loss = ", (1 / p) * np.linalg.norm((np.eye(p) - eigvec_psa @ np.diag(eigval_psa) @ eigvec_psa.T), ord='fro')**2)
    # print("Num params = ", kappa(sep_to_type(np.diff(eigval_psa))))
    # print("Eigenvalues = ", eigval_psa)

    # Eigengap sparsity
    print("EIGENGAP SPARSITY")
    start = time()
    eigval_escp, steps = pgd(X, lr=0.01, max_iter=1000, tol=1e-6)
    print("Time = ", time() - start)
    print("Objective = ", penalized_normal_loglikelihood_l0(eigval_escp, eigval_scm))
    print("Objective L1 = ", penalized_normal_loglikelihood_l1(eigval_escp, eigval_scm))
    print("Frob. loss = ", (1 / p) * np.linalg.norm((np.eye(p) - eigvec_scm @ np.diag(eigval_escp) @ eigvec_scm.T), ord='fro')**2)
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_escp))))
    print("Eigenvalues = ", eigval_escp)
    plt.figure()
    plt.plot(steps)
    plt.show(block=True)
