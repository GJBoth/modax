import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y, as_float_array
from scipy.linalg import pinvh
from scipy.linalg import solve_triangular
from numpy.linalg import LinAlgError
import warnings


def update_precisions(Q, S, q, s, A, active, tol, n_samples):
    """
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    """
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])

    # identify features that can be added , recomputed and deleted in model
    theta = q ** 2 - s
    add = (theta > 0) * (active == False)
    recompute = (theta > 0) * (active == True)
    delete = ~(add + recompute)

    # compute sparsity & quality parameters corresponding to features in
    # three groups identified above
    Qadd, Sadd = Q[add], S[add]
    Qrec, Srec, Arec = Q[recompute], S[recompute], A[recompute]
    Qdel, Sdel, Adel = Q[delete], S[delete], A[delete]

    # compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    Anew = s[recompute] ** 2 / (theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha = 1.0 / Anew - 1.0 / Arec

    # compute change in log marginal likelihood
    deltaL[add] = (Qadd ** 2 - Sadd) / Sadd + np.log(Sadd / Qadd ** 2)
    deltaL[recompute] = Qrec ** 2 / (Srec + 1.0 / delta_alpha) - np.log(
        1 + Srec * delta_alpha
    )
    deltaL[delete] = Qdel ** 2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL = deltaL / n_samples

    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)

    # no deletions or additions
    same_features = np.sum(theta[~recompute] > 0) == 0

    # changes in precision for features already in model is below threshold
    no_delta = np.sum(abs(Anew - Arec) > tol) == 0

    # check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        return [A, converged]

    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index] ** 2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            active[feature_index] = False
            A[feature_index] = np.PINF

    return [A, converged]


###############################################################################
#                ARD REGRESSION AND CLASSIFICATION
###############################################################################


# -------------------------- Regression ARD ------------------------------------


class RegressionARD:
    def __init__(self, n_iter=300, tol=1e-3):
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fits ARD Regression with Sequential Sparse Bayes Algorithm.
        
        Parameters
        -----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors
        XY = np.dot(X.T, y)
        XX = np.dot(X.T, X)
        XXd = np.diag(XX)

        #  initialise precision of noise & and coefficients
        var_y = np.var(y)

        # check that variance is non zero !!!
        if var_y == 0:
            beta = 1e-2
        else:
            beta = 1.0 / np.var(y)

        A = np.PINF * np.ones(n_features)
        active = np.zeros(n_features, dtype=np.bool)

        # in case of almost perfect multicollinearity between some features
        # start from feature 0
        if np.sum(XXd < np.finfo(np.float32).eps) > 0:
            A[0] = np.finfo(np.float16).eps
            active[0] = True
        else:
            # start from a single basis vector with largest projection on targets
            proj = XY ** 2 / XXd
            start = np.argmax(proj)
            active[start] = True
            A[start] = XXd[start] / (proj[start] - var_y)

        warning_flag = 0
        for i in range(self.n_iter):
            XXa = XX[active, :][:, active]
            XYa = XY[active]
            Aa = A[active]

            # mean & covariance of posterior distribution
            Mn, Ri, cholesky = self._posterior_dist(Aa, beta, XXa, XYa)
            if cholesky:
                Sdiag = np.sum(Ri ** 2, 0)
            else:
                Sdiag = np.copy(np.diag(Ri))
                warning_flag += 1

            # raise warning in case cholesky failes
            if warning_flag == 1:
                warnings.warn(
                    (
                        "Cholesky decomposition failed ! Algorithm uses pinvh, "
                        "which is significantly slower, if you use RVR it "
                        "is advised to change parameters of kernel"
                    )
                )

            # compute quality & sparsity parameters
            s, q, S, Q = self._sparsity_quality(
                XX, XXd, XY, XYa, Aa, Ri, active, beta, cholesky
            )

            # update precision parameter for noise distribution
            rss = np.sum((y - np.dot(X[:, active], Mn)) ** 2)
            beta = n_samples - np.sum(active) + np.sum(Aa * Sdiag)
            beta /= rss + np.finfo(np.float32).eps

            # update precision parameters of coefficients
            A, converged = update_precisions(Q, S, q, s, A, active, self.tol, n_samples)
            if converged or i == self.n_iter - 1:
                break

        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa, XYa, Aa = XX[active, :][:, active], XY[active], A[active]
        Mn, Sn, cholesky = self._posterior_dist(Aa, beta, XXa, XYa, True)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_ = Sn
        self.active_ = active
        self.lambda_ = A
        self.alpha_ = beta
        return self

    def _posterior_dist(self, A, beta, XX, XY, full_covar=False):
        """
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        """
        # compute precision matrix for active features
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        cholesky = True
        # try cholesky, if it fails go back to pinvh
        try:
            # find posterior mean : R*R.T*mean = beta*X.T*Y
            # solve(R*z = beta*X.T*Y) => find z => solve(R.T*mean = z) => find mean
            R = np.linalg.cholesky(Sinv)
            Z = solve_triangular(R, beta * XY, check_finite=False, lower=True)
            Mn = solve_triangular(R.T, Z, check_finite=False, lower=False)

            # invert lower triangular matrix from cholesky decomposition
            Ri = solve_triangular(R, np.eye(A.shape[0]), check_finite=False, lower=True)
            if full_covar:
                Sn = np.dot(Ri.T, Ri)
                return Mn, Sn, cholesky
            else:
                return Mn, Ri, cholesky
        except LinAlgError:
            cholesky = False
            Sn = pinvh(Sinv)
            Mn = beta * np.dot(Sinv, XY)
            return Mn, Sn, cholesky

    def _sparsity_quality(self, XX, XXd, XY, XYa, Aa, Ri, active, beta, cholesky):
        """
        Calculates sparsity and quality parameters for each feature
        
        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution 
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * Sn * X'
        """
        bxy = beta * XY
        bxx = beta * XXd
        if cholesky:
            # here Ri is inverse of lower triangular matrix obtained from cholesky decomp
            xxr = np.dot(XX[:, active], Ri.T)
            rxy = np.dot(Ri, XYa)
            S = bxx - beta ** 2 * np.sum(xxr ** 2, axis=1)
            Q = bxy - beta ** 2 * np.dot(xxr, rxy)
        else:
            # here Ri is covariance matrix
            XXa = XX[:, active]
            XS = np.dot(XXa, Ri)
            S = bxx - beta ** 2 * np.sum(XS * XXa, 1)
            Q = bxy - beta ** 2 * np.dot(XS, XYa)
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S), so if A = np.PINF q = Q, s = S
        qi = np.copy(Q)
        si = np.copy(S)
        #  If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa, Sa = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)
        return [si, qi, S, Q]

