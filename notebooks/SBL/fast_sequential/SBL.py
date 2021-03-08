import numpy as np
from scipy.linalg import solve_triangular
from sklearn.utils import check_X_y


def update_precisions(Q, S, q, s, alpha, tol):
    """
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    """
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])

    # identify features that can be added , recomputed and deleted in model
    theta = q ** 2 - s
    add = (theta > 0) * np.isinf(alpha)
    recompute = (theta > 0) * ~np.isinf(alpha)
    delete = ~(add + recompute)

    # compute sparsity & quality parameters corresponding to features in
    # three groups identified above
    Qadd, Sadd = Q[add], S[add]
    Qrec, Srec, alpharec = Q[recompute], S[recompute], alpha[recompute]
    Qdel, Sdel, alphadel = Q[delete], S[delete], alpha[delete]

    # compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    alphanew = s[recompute] ** 2 / (theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha = 1.0 / alphanew - 1.0 / alpharec

    # compute change in log marginal likelihood
    deltaL[add] = (Qadd ** 2 - Sadd) / Sadd + np.log(Sadd / Qadd ** 2)
    deltaL[recompute] = Qrec ** 2 / (Srec + 1.0 / delta_alpha) - np.log(
        1 + Srec * delta_alpha
    )
    deltaL[delete] = Qdel ** 2 / (Sdel - alphadel) - np.log(1 - Sdel / alphadel)

    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)

    # no deletions or additions
    same_features = np.sum(theta[~recompute] > 0) == 0

    # changes in precision for features already in model is below threshold
    no_delta = np.sum(abs(alphanew - alpharec) > tol) == 0

    # check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
    active = ~np.isinf(alpha)
    if same_features and no_delta:
        converged = True
        return [alpha, converged]

    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        alpha[feature_index] = s[feature_index] ** 2 / theta[feature_index]
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            alpha[feature_index] = np.inf
    return alpha, converged


class RegressionARD:
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
        active = ~np.isinf(alpha)
        n_samples, n_features = X.shape
        Mn, Ri = self.posterior_dist(alpha, beta, gram, XT_y)
        Sn = np.dot(Ri.T, Ri)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_ = Sn
        self.active_ = active
        self.lambda_ = alpha
        self.alpha_ = beta
        return self

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
        Ri = solve_triangular(R, np.eye(np.sum(active)), check_finite=False, lower=True)
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
        X, y = check_X_y(X, y, dtype=np.float64)  # apparently need 64 bits for log.

        gram = np.dot(X.T, X)
        XT_y = np.dot(X.T, y)

        #  initialise precision of noise & and coefficients
        beta = 1.0 / (np.var(y) + 1e-6)
        alpha = np.inf * np.ones(n_features)

        # start from a single basis vector with largest projection on targets
        proj = XT_y ** 2 / np.diag(gram)
        start = np.argmax(proj)
        alpha[start] = np.diag(gram)[start] / (proj[start] - 1 / beta)
        return alpha, beta, gram, XT_y
