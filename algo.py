import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
from scipy.optimize import isotonic_regression
from scipy.stats import ortho_group


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
    res = isotonic_regression(lbda, weights=None, increasing=False)
    return res.x  # N.B: if at some point I need the flag type, it can be obtained by `np.diff(res.blocks)`.


def pcd(loss_func, X, lr=0.01, max_iter=1000, tol=1e-6):
    steps = []
    count = 0
    V, lbda = ortho_group.rvs(dim=X.shape[0]), np.random.normal(loc=1, scale=1.0, size=X.shape[0])
    prev_loss = np.inf
    while count < max_iter:
        V = V - lr * grad(lambda V_: loss_func(lbda, V_, X))(V)
        V = project_orthogonal(V)
        lbda = lbda - lr * grad(lambda lbda_: loss_func(lbda_, V, X))(lbda)
        lbda = project_cone(lbda)
        new_loss = loss_func(lbda, V, X)
        print(new_loss)
        if (count > 0) and (abs(prev_loss - new_loss / prev_loss) < tol):  # replace with criterion on the gradient?
            break
        count += 1
        prev_loss = new_loss
    return V, lbda


if __name__ == "__main__":
    np.random.seed(42)

    n, p = 20, 10
    X = np.random.multivariate_normal(mean=np.zeros(p,), cov=np.eye(p), size=n).T  # CAUTION X must be in p \times n shape

    S = (1 / n) * X @ X.T
    eigval, eigvec = np.linalg.eigh(S)
    print(eigval, eigvec)

    V, lbda = pcd(penalized_normal_loglikelihood, X, lr=0.01, max_iter=1000, tol=1e-6)
    print(V, lbda)


