import numpy as np
import cvxpy as cvx


def markov_random_search(ret_det, returns, cov_mat, step=0.015):
    weights = np.array([1 / len(returns)] * len(returns))
    risk_min = 1  # return_max : макс доходность

    for i in range(10 ** 4):  # uniform or normal

        rand_vec = np.random.uniform(low=- step, high=0.015, size=returns.shape)
        rand_vec_centr = rand_vec - np.mean(rand_vec)  # normalization

        weights_loc = weights + rand_vec_centr

        ret_loc, risk_loc = port_return_risk(weights_loc, cov_mat.values, returns.values)

        if ret_loc > ret_det and risk_loc < risk_min and weights_loc.min() > 0.001 and weights_loc.max() < 0.5:
            weights = weights_loc
            ret_min = ret_loc

    return weights


def markov_convex_optim(ret_det, returns, cov_mat):
    n = len(mu)
    x = cvx.Variable(n, nonneg=True)
    ones_vec = np.ones(n)

    obj = cvx.Minimize(cvx.matrix_frac(x, Sigma))
    constraints = [mu.T @ x >= ret_det,
                   np.ones(n).T @ x == 1]

    prob = cvx.Problem(obj, constraints)
    prob.solve()

    weights = x.value

    return weights


class MarkovPortfolio:
    def __init__(self, returns, cov_mat, args):
        self.returns = returns
        self.cov_mat = cov_mat
        self.ret_det = args['ret_det']

    def fit(self):
        n = len(self.returns)
        x = cvx.Variable(n, nonneg=True)

        obj = cvx.Minimize(cvx.matrix_frac(x, self.cov_mat))
        constraints = [self.returns.T @ x >= self.ret_det,
                       np.ones(n).T @ x == 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        weights = x.value
        return weights


class TobinPortfolio:
    def __init__(self, returns, cov_mat, args):
        self.returns = returns
        self.cov_mat = cov_mat
        self.risk_det = args['risk_det']

    def fit(self):
        n = len(self.returns)
        x = cvx.Variable(n, nonneg=True)

        obj = cvx.Maximize(self.returns.T @ x)
        constraints = [cvx.matrix_frac(x, self.cov_mat) <= np.sqrt(self.risk_det),
                       np.ones(n).T @ x == 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        weights = x.value
        return weights


class SharpPortfolio:
    def __init__(self, returns, cov_mat, args):
        self.returns = returns
        self.cov_mat = cov_mat
        self.ret_det = args['ret_det']

    def fit(self):
        n = len(self.returns)
        x = cvx.Variable(n, nonneg=True)
        ones_vec = np.ones(n)

        obj = cvx.Maximize((self.returns.T @ x - self.ret_det) / cvx.matrix_frac(x, self.cov_mat) ** 0.5)
        constraints = [np.ones(n).T @ x == 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        weights = x.value
        return weights


class TreynorPortfolio:
    def __init__(self, returns, covar_vec, sigma_m, ret_det):
        self.returns = returns
        self.covar_vec = covar_vec
        self.sigma_m = sigma_m
        self.ret_det = ret_det

    def fit(self):
        n = len(self.returns)
        x = cvx.Variable(n, nonneg=True)

        obj = cvx.Maximize((self.returns.T @ x - self.ret_det) * self.sigma_m / (self.covar_vec.T @ x))
        constraints = [np.ones(n).T @ x == 1,
                       np.eye(n) @ x <= 0.3]

        prob = cvx.Problem(obj, constraints)
        prob.solve(qcp=True)

        weights = x.value
        return weights


class JensenPortfolio:
    def __init__(self, returns, covar_vec, sigma_m, ret_det):
        self.returns = returns
        self.covar_vec = covar_vec
        self.sigma_m = sigma_m
        self.ret_det = ret_det

    def fit(self):
        n = len(self.returns)
        x = cvx.Variable(n, nonneg=True)

        obj = cvx.Maximize(
            self.returns.T @ x - (self.ret_det + (self.returns.T - self.ret_det) * self.covar_vec.T @ x / self.sigma_m))
        constraints = [np.ones(n).T @ x == 1,
                       np.eye(n) @ x <= 0.3]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        weights = x.value
        return weights