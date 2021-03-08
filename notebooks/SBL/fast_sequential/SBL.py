import numpy as np
from scipy.linalg import solve_triangular
from sklearn.utils import check_X_y


def update_precisions(Q, S, q, s, alpha, tol):
    """
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    """
    # compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    theta = q ** 2 - s
    alphanew = s ** 2 / theta
    delta_alpha = 1.0 / alphanew - 1.0 / alpha

    # compute change in log marginal likelihood
    L_add = (Q ** 2 - S) / S + np.log(S / Q ** 2)
    L_update = Q ** 2 / (S + 1 / delta_alpha) - np.log(1 + S * delta_alpha)
    L_delete = Q ** 2 / (S - alpha) - np.log(1 - S / alpha)

    # Adding and filtering
    delta_L = np.stack([L_add, L_update, L_delete], axis=-1)
    delta_L[~np.isinf(alpha), 0] = 0.0
    delta_L[np.isinf(alpha), 1:] = 0.0
    delta_L[theta < 0, 1] = 0.0

    feature_idx, op = np.unravel_index(np.argmax(delta_L), delta_L.shape)

    # Updating alpha
    if (op == 0) or (op == 1):
        alpha[feature_idx] = alphanew[feature_idx]
    else:
        alpha[feature_idx] = np.inf

    # Checking convergence
    no_add = np.all(theta[np.isinf(alpha)] < 0)
    no_del = np.all(theta[~np.isinf(alpha)] > 0)
    max_delta_alpha = np.max(delta_alpha[~np.isinf(alpha)])
    if no_add and no_del and max_delta_alpha < tol:
        converged = True
    else:
        converged = False

    return alpha, converged


class SBL:
    def __init__(self, n_iter=300, tol=1e-3):
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        alpha, beta, gram, XT_y = self.init(X, y)
        for i in range(self.n_iter):
            Mn, Ri = self.posterior_dist(alpha, beta, gram, XT_y)
            s, q, S, Q = self.sparsity_quality(alpha, beta, gram, XT_y, Ri)
            beta = self.noise(X, y, alpha, Mn, Ri)
            alpha, converged = update_precisions(Q, S, q, s, alpha, self.tol)
            if converged or i == self.n_iter:
                break

        # after last update of alpha & beta update parameters
        # of posterior distribution
        n_samples, n_features = X.shape
        Mn, Ri = self.posterior_dist(alpha, beta, gram, XT_y)
        Sn = np.dot(Ri.T, Ri)
        active = ~np.isinf(alpha)
        coeffs = np.zeros(n_features)
        coeffs[active] = Mn
        return alpha, beta, coeffs

    def posterior_dist(self, alpha, beta, gram, XT_y):
        """
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        """
        # compute precision matrix for active features
        active = ~np.isinf(alpha)
        Sinv = beta * gram[active, :][:, active]
        np.fill_diagonal(Sinv, np.diag(Sinv) + alpha[active])

        # find posterior mean : R*R.T*mean = beta*X.T*Y
        # solve(R*z = beta*X.T*Y) => find z => solve(R.T*mean = z) => find mean
        R = np.linalg.cholesky(Sinv)
        Z = solve_triangular(R, beta * XT_y[active], check_finite=False, lower=True)
        Mn = solve_triangular(R.T, Z, check_finite=False, lower=False)

        # invert lower triangular matrix from cholesky decomposition
        Ri = solve_triangular(
            R, np.eye(np.sum(active), dtype=np.float32), check_finite=False, lower=True
        )
        return Mn, Ri

    def sparsity_quality(self, alpha, beta, gram, XT_y, Ri):
        # here Ri is inverse of lower triangular matrix obtained from cholesky decomp
        active = ~np.isinf(alpha)

        xxr = np.dot(gram[:, active], Ri.T)
        rxy = np.dot(Ri, XT_y[active])
        S = beta * np.diag(gram) - beta ** 2 * np.sum(xxr ** 2, axis=1)
        Q = beta * XT_y - beta ** 2 * np.dot(xxr, rxy)

        si, qi = np.copy(S), np.copy(Q)
        np.putmask(qi, active, alpha[active] * Q[active] / (alpha[active] - S[active]))
        np.putmask(si, active, alpha[active] * S[active] / (alpha[active] - S[active]))

        return si, qi, S, Q

    def noise(self, X, y, alpha, Mn, Ri):
        active = ~np.isinf(alpha)
        n_samples, n_features = X.shape

        Sdiag = np.sum(Ri ** 2, 0)
        rss = np.sum((y - np.dot(X[:, active], Mn)) ** 2)
        beta = n_samples - n_features + np.sum(alpha[active] * Sdiag)
        beta /= rss
        return beta

    def init(self, X, y):
        n_samples, n_features = X.shape
        X, y = check_X_y(
            X, y, dtype=np.float32, y_numeric=True
        )  # apparently need 64 bits for log.

        gram = np.dot(X.T, X)
        XT_y = np.dot(X.T, y)

        #  initialise precision of noise & and coefficients
        beta = np.float32(1.0 / (np.var(y) + 1e-6))
        alpha = np.inf * np.ones(n_features, dtype=np.float32)

        # start from a single basis vector with largest projection on targets
        proj = XT_y ** 2 / np.diag(gram)
        start = np.argmax(proj)
        alpha[start] = np.diag(gram)[start] / (proj[start] - 1 / beta)
        return alpha, beta, gram, XT_y
