import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import ledoit_wolf
from time import time

from algo import pgd
from psa import candidate_models, model_selection


if __name__ == "__main__":
    np.random.seed(42)

    n = 2000
    n_exp = 10
    n_p = 20
    times_ = np.zeros((n_exp, n_p, 4))
    for r in range(n_exp):
        print(r)
        times = []
        for p in np.logspace(1, 3, n_p).astype('int'):
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
            stepsize_init = (eigval_scm[0] - eigval_scm[-1]) / (eigval_scm[0] * alpha)
            eigval_escp, steps = pgd(X, alpha, tau=0.8, c=0.1, stepsize_init=stepsize_init, max_iter=1000, tol=1e-6)
            times_p.append(time() - start)

            times.append(times_p)

        times_[r] = np.array(times)
    plt.figure()
    plt.plot(np.logspace(1, 3, n_p).astype('int'), np.mean(times_[:, :, 0], axis=0), ls=(0, (5, 1)), color='tab:red', label="SCM")
    plt.plot(np.logspace(1, 3, n_p).astype('int'), np.mean(times_[:, :, 1], axis=0), ls=(0, (5, 5)), color='tab:orange', label="LW")
    plt.plot(np.logspace(1, 3, n_p).astype('int'), np.mean(times_[:, :, 2], axis=0), ls=(0, (5, 10)), color='tab:blue', label="PSA")
    plt.plot(np.logspace(1, 3, n_p).astype('int'), np.mean(times_[:, :, 3], axis=0), ls=(5, (10, 3)), color='tab:green', label="ESCP")
    plt.xscale('log')
    plt.legend()
    plt.show()