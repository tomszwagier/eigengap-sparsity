import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from scipy.optimize import isotonic_regression
from sklearn.covariance import ledoit_wolf

from psa import candidate_models, model_selection, sep_to_type, kappa


def penalized_normal_loglikelihood_l0(eigval, eigval_scm, alpha, n):
    return np.sum(np.log(eigval) + (eigval_scm / eigval)) + (alpha / n) * kappa(sep_to_type(np.diff(eigval)))


def penalized_normal_loglikelihood_l1_rel(eigval, eigval_scm, alpha, n):
    return np.sum(np.log(eigval) + (eigval_scm / eigval)) + (alpha / n) * (-np.sum(eigval[1:]/eigval[:-1]) - np.sum(np.tril(eigval[:, None] / eigval, k=-1)))


def project_cone(eigval):
    res = isotonic_regression(eigval, weights=None, increasing=False)
    return np.clip(res.x, 1e-10, 1e10)


def pgd(X, alpha=0, tau=0.8, c=0.1, stepsize_init=0.01, max_iter=1000, tol=1e-6):
    p, n = X.shape
    steps = []
    count = 0
    S = (1 / n) * X @ X.T
    eigval_scm, _ = np.linalg.eigh(S)
    eigval_scm = eigval_scm[::-1]
    eigval = eigval_scm
    prev_loss = np.inf
    f = lambda eigval_: penalized_normal_loglikelihood_l1_rel(eigval_, eigval_scm, alpha, n)
    while count < max_iter:
        stepsize = stepsize_init
        g = grad(f)(eigval)
        eigval_new = project_cone(eigval - stepsize * g)
        while f(eigval_new) > f(eigval) + c * np.dot(g, eigval_new - eigval):
            stepsize = stepsize * tau
            eigval_new = project_cone(eigval - stepsize * g)
        eigval = eigval_new
        new_loss = penalized_normal_loglikelihood_l1_rel(eigval, eigval_scm, alpha, n)
        steps.append(new_loss)
        if (count > 0) and (abs((prev_loss - new_loss) / prev_loss) < tol):
            break
        count += 1
        prev_loss = new_loss
    return eigval, steps


def frob_loss(cov_true, cov_est):
    p = cov_true.shape[0]
    return (1 / p) * np.linalg.norm((cov_true - cov_est), ord='fro')**2


if __name__ == "__main__":
    np.random.seed(42)

    n_exp = 10
    results = np.zeros((n_exp, 4, 3))
    for r in range(n_exp):

        # Dataset
        n, p = 200, 100  # 40, 20 / 200, 100
        cov_true = np.eye(p)
        X = np.random.multivariate_normal(mean=np.zeros(p,), cov=cov_true, size=n).T
        # n, p = 400, 200
        # cov_true = np.diag([10] * int(.4 * p) + [1] * int(.4 * p) + [.1] * int(.2 * p))
        # X = np.random.multivariate_normal(mean=np.zeros(p,), cov=cov_true, size=n).T
        alpha = np.log(n)  # BIC

        # Sample covariance
        print("SAMPLE COVARIANCE")
        S = (1 / n) * X @ X.T
        eigval_scm, eigvec_scm = np.linalg.eigh(S)
        eigval_scm, eigvec_scm = eigval_scm[::-1], eigvec_scm[:, ::-1]
        print("Pen. Lik. = ", penalized_normal_loglikelihood_l0(eigval_scm, eigval_scm, alpha, n))
        print("Frob. loss = ", frob_loss(cov_true, S))
        print("Num params = ", kappa(sep_to_type(np.diff(eigval_scm))))
        results[r, 0] = np.array([penalized_normal_loglikelihood_l0(eigval_scm, eigval_scm, alpha, n), frob_loss(cov_true, S), kappa(sep_to_type(np.diff(eigval_scm)))])

        # Ledoit-Wolf
        print("LEDOIT-WOLF")
        shrunk_cov, shrinkage = ledoit_wolf(X.T, assume_centered=True)
        eigval_lw, _ = np.linalg.eigh(shrunk_cov)
        eigval_lw = eigval_lw[::-1]
        print("Pen. Lik. = ", penalized_normal_loglikelihood_l0(eigval_lw, eigval_scm, alpha, n))
        print("Frob. loss = ", frob_loss(cov_true, shrunk_cov))
        print("Num params = ", kappa(sep_to_type(np.diff(eigval_lw))))
        results[r, 1] = np.array([penalized_normal_loglikelihood_l0(eigval_lw, eigval_scm, alpha, n), frob_loss(cov_true, shrunk_cov), kappa(sep_to_type(np.diff(eigval_lw)))])

        # Principal subspace analysis
        if p <= 20:
            print("PRINCIPAL SUBSPACE ANALYSIS")
            models = candidate_models(p)
            model_best, eigval_psa, eigvec_psa = model_selection(X, models)
            print("Pen. Lik. = ", penalized_normal_loglikelihood_l0(eigval_psa, eigval_scm, alpha, n))
            print("Frob. loss = ", frob_loss(cov_true, eigvec_psa @ np.diag(eigval_psa) @ eigvec_psa.T))
            print("Num params = ", kappa(sep_to_type(np.diff(eigval_psa))))
            results[r, 2] = np.array([penalized_normal_loglikelihood_l0(eigval_psa, eigval_scm, alpha, n), frob_loss(cov_true, eigvec_psa @ np.diag(eigval_psa) @ eigvec_psa.T), kappa(sep_to_type(np.diff(eigval_psa)))])

        # Eigengap sparsity
        print("EIGENGAP SPARSITY")
        stepsize_init = (eigval_scm[0] - eigval_scm[-1]) / (eigval_scm[0] * alpha)
        eigval_escp, steps = pgd(X, alpha, tau=0.8, c=0.1, stepsize_init=stepsize_init, max_iter=1000, tol=1e-6)  # 50000 / 1e-10 for the last example
        print("Pen. Lik. = ", penalized_normal_loglikelihood_l0(eigval_escp, eigval_scm, alpha, n))
        print("Frob. loss = ", frob_loss(cov_true, eigvec_scm @ np.diag(eigval_escp) @ eigvec_scm.T))
        print("Num params = ", kappa(sep_to_type(np.diff(eigval_escp))))
        results[r, 3] = np.array([penalized_normal_loglikelihood_l0(eigval_escp, eigval_scm, alpha, n), frob_loss(cov_true, eigvec_scm @ np.diag(eigval_escp) @ eigvec_scm.T), kappa(sep_to_type(np.diff(eigval_escp)))])
        plt.figure()
        plt.plot(steps)
        plt.show()

        plt.figure()
        plt.plot(np.diag(cov_true), ls="solid", color='k', label="True")
        plt.plot(eigval_scm, ls=(0, (5, 1)), color='tab:red', label="SCM")
        plt.plot(eigval_lw, ls=(0, (5, 5)), color='tab:orange', label="LW")
        if p <= 20:
            plt.plot(eigval_psa, ls=(0, (5, 10)), color='tab:blue', label="PSA")
        plt.plot(eigval_escp, ls=(5, (10, 3)), color='tab:green', label="ESCP")
        plt.legend()
        plt.show()

    print(np.mean(results, axis=0))
    print(np.std(results, axis=0))
