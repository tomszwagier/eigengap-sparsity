import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from scipy.optimize import isotonic_regression
from sklearn.covariance import ledoit_wolf
from time import time

from psa import candidate_models, model_selection, sep_to_type, kappa


def penalized_normal_loglikelihood_l0(eigval, eigval_scm, alpha, n):
    return np.sum(np.log(eigval) + (eigval_scm / eigval)) + (alpha / n) * kappa(sep_to_type(np.diff(eigval)))


def penalized_normal_loglikelihood_l1_rel(eigval, eigval_scm, alpha, n):
    return np.sum(np.log(eigval) + (eigval_scm / eigval)) + (alpha / n) * (-np.sum(eigval[1:]/eigval[:-1]) - np.sum(np.tril(eigval[:, None] / eigval, k=-1)))


def project_cone(eigval):
    res = isotonic_regression(eigval, weights=None, increasing=False)
    return np.clip(res.x, 1e-10, 1e10)


def pgd(X, alpha=0, lr=0.01, max_iter=1000, tol=1e-6):
    p, n = X.shape
    steps = []
    count = 0
    S = (1 / n) * X @ X.T
    eigval_scm, _ = np.linalg.eigh(S)
    eigval_scm = eigval_scm[::-1]
    eigval = np.copy(eigval_scm)
    prev_loss = np.inf
    while count < max_iter:
        eigval = eigval - lr * (grad(lambda eigval_: penalized_normal_loglikelihood_l1_rel(eigval_, eigval_scm, alpha, n))(eigval))
        eigval = project_cone(eigval)
        new_loss = penalized_normal_loglikelihood_l1_rel(eigval, eigval_scm, alpha, n)
        steps.append(new_loss)
        if (count > 0) and (abs((prev_loss - new_loss) / prev_loss) < tol):
            break
        count += 1
        prev_loss = new_loss
        # print(sep_to_type(np.diff(eigval)))
    return eigval, steps


def frob_loss(cov_true, cov_est):
    p = cov_true.shape[0]
    return (1 / p) * np.linalg.norm((cov_true - cov_est), ord='fro')**2


if __name__ == "__main__":
    np.random.seed(42)

    n, p = 200, 100  # 40, 20 / 200, 100
    cov_true = np.eye(p)
    X = np.random.multivariate_normal(mean=np.zeros(p,), cov=cov_true, size=n).T
    # n, p = 800, 200
    # cov_true = np.diag([10] * int(.4 * p) + [3] * int(.4 * p) + [1] * int(.2 * p))
    # X = np.random.multivariate_normal(mean=np.zeros(p,), cov=cov_true, size=n).T
    alpha = np.log(n)  # BIC

    # Sample covariance
    print("SAMPLE COVARIANCE")
    start = time()
    S = (1 / n) * X @ X.T
    eigval_scm, eigvec_scm = np.linalg.eigh(S)
    eigval_scm, eigvec_scm = eigval_scm[::-1], eigvec_scm[:, ::-1]
    print("Time = ", time() - start)
    print("Objective L0 = ", penalized_normal_loglikelihood_l0(eigval_scm, eigval_scm, alpha, n))
    print("Objective L1-REL = ", penalized_normal_loglikelihood_l1_rel(eigval_scm, eigval_scm, alpha, n))
    print("Frob. loss = ", frob_loss(cov_true, S))
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_scm))))
    # print("Eigenvalues = ", eigval_scm)

    # Ledoit-Wolf
    print("LEDOIT-WOLF")
    start = time()
    shrunk_cov, shrinkage = ledoit_wolf(X.T, assume_centered=True)
    eigval_lw, _ = np.linalg.eigh(shrunk_cov)
    eigval_lw = eigval_lw[::-1]
    print("Time = ", time() - start)
    print("Objective L0 = ", penalized_normal_loglikelihood_l0(eigval_lw, eigval_scm, alpha, n))
    print("Objective L1-REL = ", penalized_normal_loglikelihood_l1_rel(eigval_lw, eigval_scm, alpha, n))
    print("Frob. loss = ", frob_loss(cov_true, shrunk_cov))
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_lw))))
    # print("Eigenvalues = ", eigval_lw)

    # # Principal subspace analysis
    # print("PRINCIPAL SUBSPACE ANALYSIS")
    # start = time()
    # models = candidate_models(p)
    # model_best, eigval_psa, eigvec_psa = model_selection(X, models)
    # print("Time = ", time() - start)
    # print("Objective L0 = ", penalized_normal_loglikelihood_l0(eigval_psa, eigval_scm, alpha, n))
    # print("Objective L1-REL = ", penalized_normal_loglikelihood_l1_rel(eigval_psa, eigval_scm, alpha, n))
    # print("Frob. loss = ", frob_loss(cov_true, eigvec_psa @ np.diag(eigval_psa) @ eigvec_psa.T))
    # print("Num params = ", kappa(sep_to_type(np.diff(eigval_psa))))
    # # print("Eigenvalues = ", eigval_psa)

    # Eigengap sparsity
    print("EIGENGAP SPARSITY")
    start = time()
    eigval_escp, steps = pgd(X, alpha, lr=0.1, max_iter=1000, tol=1e-6)
    print("Time = ", time() - start)
    print("Objective L0 = ", penalized_normal_loglikelihood_l0(eigval_escp, eigval_scm, alpha, n))
    print("Objective L1 REL = ", penalized_normal_loglikelihood_l1_rel(eigval_escp, eigval_scm, alpha, n))
    print("Frob. loss = ", frob_loss(cov_true, eigvec_scm @ np.diag(eigval_escp) @ eigvec_scm.T))
    print("Num params = ", kappa(sep_to_type(np.diff(eigval_escp))))
    # print("Eigenvalues = ", eigval_escp)
    plt.figure()
    plt.plot(steps)
    plt.show()

    plt.figure()
    plt.plot(np.diag(cov_true), ls="solid", color='k', label="True")
    plt.plot(eigval_scm, ls=(0, (5, 1)), color='tab:red', label="SCM")
    plt.plot(eigval_lw, ls=(0, (5, 5)), color='tab:orange', label="LW")
    # plt.plot(eigval_psa, ls=(0, (5, 10)), color='tab:blue', label="PSA")
    plt.plot(eigval_escp, ls=(5, (10, 3)), color='tab:green', label="ESCP")
    plt.legend()
    plt.show()


    ### TIME CURVES
    n = 2000
    times_ = np.zeros((10, 20, 4))
    for r in range(10):
        print(r)
        times = []
        for p in np.logspace(1, 3, 20).astype('int'):
            cov_true = np.eye(p)
            X = np.random.multivariate_normal(mean=np.zeros(p,), cov=cov_true, size=n).T
            alpha = np.log(n)
            times_p = []

            # Sample covariance
            start = time()
            S = (1 / n) * X @ X.T
            eigval_scm, eigvec_scm = np.linalg.eigh(S)
            eigval_scm, eigvec_scm = eigval_scm[::-1], eigvec_scm[:, ::-1]
            times_p.append(time() - start)

            # Ledoit-Wolf
            start = time()
            shrunk_cov, shrinkage = ledoit_wolf(X.T, assume_centered=True)
            eigval_lw, _ = np.linalg.eigh(shrunk_cov)
            eigval_lw = eigval_lw[::-1]
            times_p.append(time() - start)

            # Principal subspace analysis
            if p < 20:
                start = time()
                models = candidate_models(p)
                model_best, eigval_psa, eigvec_psa = model_selection(X, models)
                times_p.append(time() - start)
            else:  # will be too long, so let's say the time after p=20 is above 50s for this plot (which is what we observe in practice)
                times_p.append(50)

            # Eigengap sparsity
            start = time()
            eigval_escp, steps = pgd(X, alpha, lr=0.1, max_iter=1000, tol=1e-6)
            times_p.append(time() - start)

            times.append(times_p)

        times_[r] = np.array(times)
    plt.figure()
    plt.plot(np.logspace(1, 3, 20).astype('int'), np.mean(times_[:, :, 0], axis=0), ls=(0, (5, 1)), color='tab:red', label="SCM")
    plt.plot(np.logspace(1, 3, 20).astype('int'), np.mean(times_[:, :, 1], axis=0), ls=(0, (5, 5)), color='tab:orange', label="LW")
    plt.plot(np.logspace(1, 3, 20).astype('int'), np.mean(times_[:, :, 2], axis=0), ls=(0, (5, 10)), color='tab:blue', label="PSA")
    plt.plot(np.logspace(1, 3, 20).astype('int'), np.mean(times_[:, :, 3], axis=0), ls=(5, (10, 3)), color='tab:green', label="ESCP")
    plt.xscale('log')
    plt.legend()
    plt.show()
