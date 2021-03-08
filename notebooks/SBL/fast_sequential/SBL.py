import numpy as np
from scipy.linalg import solve_triangular
from sklearn.utils import check_X_y


def update_precisions(Q, S, q, s, alpha, tol, n_samples):
    """
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    """
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])

    # identify features that can be added , recomputed and deleted in model
    active = ~np.isinf(alpha)
    theta = q ** 2 - s
    add = (theta > 0) * (active == False)
    recompute = (theta > 0) * (active == True)
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
    deltaL = deltaL / n_samples

    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)

    # no deletions or additions
    same_features = np.sum(theta[~recompute] > 0) == 0

    # changes in precision for features already in model is below threshold
    no_delta = np.sum(abs(alphanew - alpharec) > tol) == 0

    # check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
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
        n_samples, n_features = X.shape
        X, y = check_X_y(X, y, dtype=np.float64)  # apparently need 64 bits for log.

        XY = np.dot(X.T, y)
        XX = np.dot(X.T, X)
        XXd = np.diag(XX)

        #  initialise precision of noise & and coefficients
        beta = 1.0 / (np.var(y) + 1e-6)
        alpha = np.inf * np.ones(n_features)

        # start from a single basis vector with largest projection on targets
        proj = XY ** 2 / XXd
        start = np.argmax(proj)
        alpha[start] = XXd[start] / (proj[start] - 1 / beta)

        for i in range(self.n_iter):
            active = ~np.isinf(alpha)
            XXa = XX[active, :][:, active]
            XYa = XY[active]

            # mean & covariance of posterior distribution
            Mn, Ri = self._posterior_dist(alpha[active], beta, XXa, XYa)
            Sdiag = np.sum(Ri ** 2, 0)

            # compute quality & sparsity parameters
            s, q, S, Q = self._sparsity_quality(XX, XXd, XY, XYa, alpha, Ri, beta)

            # update precision parameter for noise distribution
            rss = np.sum((y - np.dot(X[:, active], Mn)) ** 2)
            beta = n_samples - np.sum(active) + np.sum(alpha[active] * Sdiag)
            beta /= rss + np.finfo(np.float32).eps

            # update precision parameters of coefficients
            alpha, converged = update_precisions(Q, S, q, s, alpha, self.tol, n_samples)
            if converged or i == self.n_iter - 1:
                break

        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa, XYa, Aa = XX[active, :][:, active], XY[active], alpha[active]
        Mn, Sn = self._posterior_dist(Aa, beta, XXa, XYa, True)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_ = Sn
        self.active_ = active
        self.lambda_ = alpha
        self.alpha_ = beta
        return self

    def _posterior_dist(self, alpha, beta, XX, XY, full_covar=False):
        """
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        """
        # compute precision matrix for active features
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + alpha)

        # find posterior mean : R*R.T*mean = beta*X.T*Y
        # solve(R*z = beta*X.T*Y) => find z => solve(R.T*mean = z) => find mean
        R = np.linalg.cholesky(Sinv)
        Z = solve_triangular(R, beta * XY, check_finite=False, lower=True)
        Mn = solve_triangular(R.T, Z, check_finite=False, lower=False)

        # invert lower triangular matrix from cholesky decomposition
        Ri = solve_triangular(R, np.eye(alpha.shape[0]), check_finite=False, lower=True)
        if full_covar:
            Sn = np.dot(Ri.T, Ri)
            return Mn, Sn
        else:
            return Mn, Ri

    def _sparsity_quality(self, XX, XXd, XY, XYa, alpha, Ri, beta):
        """
        Calculates sparsity and quality parameters for each feature
        
        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution 
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * Sn * X'
        """

        # here Ri is inverse of lower triangular matrix obtained from cholesky decomp
        active = ~np.isinf(alpha)
        Aa = alpha[active]
        xxr = np.dot(XX[:, active], Ri.T)
        rxy = np.dot(Ri, XYa)
        S = beta * XXd - beta ** 2 * np.sum(xxr ** 2, axis=1)
        Q = beta * XY - beta ** 2 * np.dot(xxr, rxy)

        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S), so if A = np.PINF q = Q, s = S
        qi = np.copy(Q)
        si = np.copy(S)
        #  If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa, Sa = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)
        return si, qi, S, Q

