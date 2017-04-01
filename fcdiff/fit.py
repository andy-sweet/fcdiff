"""
Fits models to observed correlations.
"""

import numpy as np
import scipy.stats
import scipy.misc
import scipy.optimize

from . import util

class UnsharedRegionFit(object):
    """
    Fits an unshared region model to correlations.

    Attributes
    -------
    model : :class:`fcdiff.UnsharedRegionModel`
        Initial model (default randomly initialized using ``seed``).
    b : :class:`numpy.ndarray`, (C, H), -1 <= float <= 1
        Correlations of healthy subjects.
    bt : :class:`numpy.ndarray`, (C, U), -1 <= float <= 1
        Correlations of unhealthy patients.
    max_iters : 1 <= int
        Maximum number of iterations.
    rel_tol : 0 < float
        Relative tolerance used to determined convergence.
    energy : list< float >
        Variational free energy at each iteration.
    """

    def __init__(self):
        self.model = None
        self.b = None
        self.bt = None
        self.max_iters = 10
        self.rel_tol = 1e-5
        self.energy = []

        # _lq_R : :class:`numpy.ndarray`, (N, U, 2), float <= 0
        #     Log variational approximation of :math:`p(r | b, \tilde{b})`.
        # _lq_F : :class:`numpy.ndarray`, (C, 3), float <= 0
        #     Log of variational approximation of :math:`p(f | b, \tilde{b})`.
        # _lp_B_g_F : :class:`numpy.ndarray`, (C, H, 3), float
        #     Normal component log densities of subject correlations.
        # _p_Bt_g_Ft : :class:`numpy.ndarray`, (C, U, 3), float
        #     Normal component densities of patient correlations.
        # _lM : :class:`numpy.ndarray`, (C, U, 3, 3), float
        #     Mixture log densities of patients.
        self._lq_R = None
        self._lq_F = None
        self._lp_B_g_F = None
        self._p_Bt_g_Ft = None
        self._lM = None

    def run(self):
        """
        Runs the fitting procedure.
        """
        (C, H) = self.b.shape
        U = self.bt.shape[1]
        N = util.C_to_N(C)
        if (N % 1) != 0:
            msg = "Number of connections (%u) must be a triangular number." % C
            raise ValueError(msg)
        if self.model is None:
            msg = "Model has not been initialized."
            raise ValueError(msg)

        self._init_lps(N, H, U)
        self._update_lps()

        self.energy = []
        self.energy[0] = self._eval_energy()
        for i in range(1, self.max_iters + 1):
            self._update_lq_F()
            self._update_lq_R()
            self._update_theta()
            self._update_lps()
            self.energy[i] = self._eval_energy()
            if self._is_converged(i):
                break

    def _init_lps(self, N, H, U):
        """
        Initializes the log probabilities to be uniform.

        Arguments
        ---------
        N : 2 <= int
            Number of regions.
        H : 1 <= int
            Number of healthy subjects.
        U : 1 <= int
            Number of unhealthy patients.
        """
        C = util.N_to_C(N)
        self._lq_R = np.full((N, U, 2), -np.log(2))
        self._lq_F = np.full((C, 1, 3), -np.log(3))
        self._lp_B_g_F = np.full((C, H, 3), 1)
        self._p_Bt_g_Ft = np.full((C, U, 3), 1)
        self._lM = np.full((C, U, 3, 3), 1)

    def _update_lps(self):
        """
        Updates the log probabilities based on the current parameters.

        This function assumes that the log probabilities have already been
        initialized with _init_lps.
        """
        for k in range(3):
            mu = self.model.mu[k]
            sigma = self.model.sigma[k]
            self._lp_B_g_F[:, :, k] = scipy.stats.norm.logpdf(self.b, mu, sigma)
            self._p_Bt_g_Ft[:, :, k] = scipy.stats.norm.pdf(self.bt, mu, sigma)

        eta = self.model.eta
        epsilon = self.model.epsilon
        for k in range(3):
            for l in range(3):
                M = _eval_M(self._p_Bt_g_Ft, eta, epsilon, k, l)
                self._lM[:, :, k, l] = np.log(M)

    def _is_converged(self, s):
        """
        Checks the convergence of the minimization.

        Arguments
        ---------
        s : 1 <= int
            Current iteration number.

        Returns
        -------
        bool
            True if the minimization has converged, False otherwise.
        """
        e = self.energy[s - 1]
        e_star = self.energy[s]
        return ((e - e_star) / e) < self.rel_tol

    def _eval_energy(self):
        """
        Computes the energy.
        """
        q_F = np.exp(self._lq_F)
        q_R = np.exp(self._lq_R)
        energy = 0
        energy -= _eval_E_lp_F(q_F, self.model.gamma)
        energy -= _eval_E_lp_B_g_F(q_F, self._lp_B_g_F)
        energy -= _eval_E_lp_R(q_R, self.model.pi)
        energy -= _eval_E_lM(q_F, q_R, self._lM)
        energy += _eval_E_lq_F(q_F, self._lq_F)
        energy += _eval_E_lq_R(q_R, self._lq_R)
        return energy

    def _update_lq_F(self):
        """
        Update the probability of the typical network template.

        This assumes that self.lp_B_g_F and self.lM are up to date.
        """
        (N, U) = self._lq_R.shape[0:2]
        (C, H) = self._lp_B_g_F.shape[0:2]
        lq_F = np.tile(np.log(self.model.gamma), (C, 1, 1))
        q_R = np.exp(self._lq_R)
        for c in range(C):
            (n, m) = util.c_to_nm(c)
            q_R_w = _eval_q_R_w(q_R, n, m)
            for k in range(3):
                sum_lp_B_g_F = np.sum(self._lp_B_g_F[c, :, k])
                sum_lM = np.sum(q_R_w * self._lM[c, :, k, :])
                lq_F[c, :, k] += sum_lp_B_g_F + sum_lM
        self._lq_F = lq_F - scipy.misc.logsumexp(lq_F, axis = 2, keepdims = True)

    def _update_lq_R(self):
        """
        Update the probability of the anomalous regions.
        """
        (N, U) = self._lq_R.shape[0:2]
        q_R = np.exp(self._lq_R)
        q_F = np.exp(self._lq_F)
        lq_R = np.tile(np.log(self.model.pi), (N, U, 1))
        for n in range(N):
            for m in list(set(range(N)) - set([n])):
                c = util.nm_to_c(n, m)
                for k in range(3):
                    lM_00 = q_R[m, :, 0] * self._lM[c, :, k, 0]
                    lM_1neq = q_R[m, :, 1] * self._lM[c, :, k, 2]
                    lq_R[n, :, 0] += q_F[c, :, k] * (lM_00 + lM_1neq)

                    lM_11 = q_R[m, :, 1] * self._lM[c, :, k, 1]
                    lM_0neq = q_R[m, :, 0] * self._lM[c, :, k, 2]
                    lq_R[n, :, 1] += q_F[c, :, k] * (lM_11 + lM_0neq)

            lq_R[n, :, :] -= scipy.misc.logsumexp(lq_R[n, :, :], axis = 1, keepdims = True)
            q_R[n, :, :] = np.exp(lq_R[n, :, :])
        self._lq_R = lq_R

    def _update_theta(self):
        """
        Update the parameters of the model.
        """
        self._update_pi()
        self._update_gamma()
        self._update_theta_sub()

    def _update_pi(self):
        """
        Updates the value of the parameter pi.
        """
        q_R = np.exp(self._lq_R)
        self.model.pi = np.mean(q_R[:, :, 1])

    def _update_gamma(self):
        """
        Updates the value of the parameter gamma.
        """
        q_F = np.exp(self._lq_F)
        self.model.gamma = np.mean(q_F, axis = (0, 1))

    def _update_theta_sub(self):
        """
        Updates the values of the parameters (eta, epsilon, mu, sigma).
        """
        import scipy.optimize as spopt
        theta_sub = self._pack_theta_sub()
        eps = 1e-5
        bnds = (
            (eps, 1 - eps),
            (eps, 1 - eps),
        #    (-1 + eps, 0 - eps),
        #    (0 - eps, 0 + eps),
        #    (0 + eps, 1 - eps),
        #    (0 + eps, None),
        #    (0 + eps, None),
        #    (0 + eps, None),
        )
        opt_result = spopt.minimize(self.opt_fun, theta_sub, jac = False,
                bounds = bnds, options = {'disp': True})
        self._unpack_theta_sub(opt_result.x)

    def _pack_theta_sub(self):
        """
        Packs the subset of theta that is jointly optimized into one vector.
        """
        theta_sub_list = (
            [self.model.eta],
            [self.model.epsilon],
            #self.model.mu,
            #self.model.sigma ** 2,
        )
        return np.concatenate(theta_sub_list)

    def _unpack_theta_sub(self, theta_sub):
        """
        Unpacks the subset of theta that is jointly optimized into one vector.

        Arguments
        ---------
        theta_sub : :class:`numpy.ndarray`, (8,), float
            Sub-vector of theta containing [eta, epsilon, mu, sigma].
        """
        self.model.eta = theta_sub[0]
        self.model.epsilon = theta_sub[1]
        #self.model.mu = theta_sub[2:5]
        #self.model.sigma = np.sqrt(theta_sub[5:8])


    def _opt_fun(self, theta_sub):
        """
        Computes the energy and jacobian to find theta_rest.

        Arguments
        ---------
        theta_sub : :class:`numpy.ndarray`, (8,), float
            Sub-vector of theta containing [eta, epsilon, mu, sigma].
        """
        self._unpack_theta_sub(theta_sub)
        self._update_lps()
        energy = 0
        #energy -= self._E_lp_B_g_F()
        energy -= self._E_lM()
        return energy
        #jac = self._opt_jac()
        #return (energy, jac)


    def _opt_jac(self):
        """
        Computes the jacobian of the energy.
        """
        (N, U) = self.q_R.shape[0:2]
        C = self.q_F.shape[0]
        dE_dh = 0
        dE_de = 0
        dE_dm = np.zeros((3,))
        dE_ds = np.zeros((3,))
        eta = self.model.eta
        epsilon = self.model.epsilon
        mu = self.model.mu
        sigma = self.model.sigma
        for c in range(C):
            (n, m) = util.c_to_nm(c)
            q_R_w = _eval_q_R_w(q_R, n, m)
            for j in range(3):
                dlNj_dm = self._d_log_N_d_mu(self.b[c, :], mu[j], sigma[j])
                dE_dm[j] -= self.q_F[c, 0, j] * np.sum(dlNj_dm)
                dlNj_ds = self._d_log_N_d_sigma(self.b[c, :], mu[j], sigma[j])
                dE_ds[j] -= self.q_F[c, 0, j] * np.sum(dlNj_ds)

            for k in range(3):
                dlM_dh = self._dlM_dh(c, k)
                dlM_de = np.zeros((U, 3))
                dlM_dm = np.zeros((U, 3, 3))
                dlM_ds = np.zeros((U, 3, 3))
                for l in range(3):
                    M_ckl = M_ck[:, l]
                    dlM_de[:, l] = self._dlM_de(c, k, l)
                    for j in range(3):
                        dlM_dm[:, j, l] = self._dlM_dm(c, k, l, j)
                        dlM_ds[:, j, l] = self._dlM_ds(c, k, l, j)

                q_F_ck = self.q_F[c, 0, k]
                dE_dh -= q_F_ck * np.sum(q_R_w[:, 2] * dlM_dh)
                dE_de -= q_F_ck * np.sum(q_R_w * dlM_de)
                for j in range(3):
                    dE_dm[j] -= q_F_ck * np.sum(q_R_w * dlM_dm[:, j, :])
                    dE_ds[j] -= q_F_ck * np.sum(q_R_w * dlM_ds[:, j, :])

        jac_list = (
            [dE_dh],
            [dE_de],
            #dE_dm,
            #dE_ds,
        )
        return np.concatenate(jac_list)

    def _eval_dE_dm(self, q_R, q_F):
        """
        Evaluates :math:`\frac{\partial \mathcal{E}}{\partial \mu}`.
        """
        (N, U) = q_R.shape[0:2]
        C = q_F.shape[0]
        dE_dm = np.zeros((3,))
        eta = self.model.eta
        epsilon = self.model.epsilon
        mu = self.model.mu
        sigma = self.model.sigma
        for c in range(C):
            (n, m) = util.c_to_nm(c)
            q_R_w = _eval_q_R_w(q_R, n, m)
            for j in range(3):
                dlN_dm = _eval_dlN_dm(self.b[c, :], mu[j], sigma[j])
                dE_dm[j] -= self.q_F[c, 0, j] * np.sum(dlN_dm)
            for k in range(3):
                dlM_dm = np.zeros((U, 3, 3))
                for l in range(3):
                    M_ckl = M_ck[:, l]
                    for j in range(3):
                        dlM_dm[:, j, l] = _eval_dlM_dm(c, k, l, j)
                q_F_ck = self.q_F[c, 0, k]
                for j in range(3):
                    dE_dm[j] -= q_F_ck * np.sum(q_R_w * dlM_dm[:, j, :])


    def _eval_dlM_dm(self, c, k, l, j):
        """
        Computes the derivative of log M w.r.t. mu.
        """
        eta = self.model.eta
        epsilon = self.model.epsilon
        mu = self.model.mu
        sigma = self.model.sigma
        e = _eval_M_eps(eta, epsilon, l)
        if j != k:
            e = (1 - e) / 2
        M = self._lM[c, :, k, l]
        return e * _dN_dm(self.bt[c, :], mu[j], sigma[j]) / M


def _eval_q_R_w(q_R, n, m):
    """
    Evaluates the three weights associated with a single pair of nodes.

    Arguments
    ---------
    q_R : :class:`numpy.ndarray`, (N, U, 2), float <= 0
        Variational approximation of :math:`p(r | b, \tilde{b})`.
    n : 0 <= int <= N
        Region index to evaluate.
    m : n < int <= N
        Other region index to evaluate.

    Returns
    -------
    q_R_w : :class:`numpy.ndarray`, (U, 3), float <= 0
        Three weights associated with a single pair of nodes.
    """
    (N, U) = q_R.shape[0:2]
    q_R_w = np.zeros((U, 3))
    q_R_w[:, 0] = q_R[n, :, 0] * q_R[m, :, 0]
    q_R_w[:, 1] = q_R[n, :, 1] * q_R[m, :, 1]
    q_R_w[:, 2] = q_R[n, :, 0] * q_R[m, :, 1]
    q_R_w[:, 2] += q_R[n, :, 1] * q_R[m, :, 0]
    return q_R_w


def _eval_M(N, eta, epsilon, k, l):
    """
    Evaluates :math:`\mathcal{M}_{kl}` based on the given normal densities.

    Arguments
    ---------
    N : :class:`numpy.ndarray`, (C, U, 3), float
        The normal densities.
    k : 0 <= int <= 2
        Component index to evaluate.
    l : 0 <= int <= 2
        Component index to evaluate.

    Returns
    -------
    M : :class:`numpy.ndarray`, (C, U), float
        :math:`\mathcal{M}_{kl}(\tilde{b}; \theta)`.
    """
    eps = _eval_M_eps(eta, epsilon, l)
    js = list(set(range(3)) - set([k]))
    sum_N = np.sum(N[:, :, js], axis = 2)
    return eps * N[:, :, k] + (1 - eps) * 0.5 * sum_N


def _eval_M_eps(eta, epsilon, l):
    """
    Evaluates the probability of the same connection type in cases of M.
    """
    if l == 0:
        eps = 1 - epsilon
    elif l == 1:
        eps = epsilon
    elif l == 2:
        eps = eta * epsilon
        eps += (1 - eta) * (1 - epsilon)
    return eps


def _eval_E_lp_F(q_F, gamma):
    """
    Evaluates :math:`\mathbb{E}[\log p(f; \gamma)]`.

    Arguments
    ---------
    q_F : :class:`numpy.ndarray`, (C, 1, 3), 0 <= float <= 1
        Variational approximation of :math:`p(f | b, \tilde{b})`.
    gamma : :class:`numpy.ndarray`, (3,), 0 <= float <= 1
        Probability of each template connection type (sums to 1).
    """
    return np.sum(q_F * np.log(gamma))


def _eval_E_lp_B_g_F(q_F, lp_B_g_F):
    """
    Evaluates :math:`\mathbb{E}[\log p(b | f; \mu, \sigma)]`.

    Arguments
    ---------
    q_F : :class:`numpy.ndarray`, (C, 1, 3), 0 <= float <= 1
        Variational approximation of :math:`p(f | b, \tilde{b})`.
    lp_B_g_F : :class:`numpy.ndarray`, (C, H, 3), float
        Normal component log densities of subject correlations.
    """
    return np.sum(q_F * lp_B_g_F)


def _eval_E_lp_R(q_R, pi):
    """
    Evaluates :math:`\mathbb{E}[\log p(r; \pi)]`.

    Arguments
    ---------
    q_R : :class:`numpy.ndarray`, (N, U, 2), 0 <= float <= 1
        Variational approximation of :math:`p(r | b, \tilde{b})`.
    pi : 0 <= float <= 1
        Probability of an anomalous region.
    """
    return np.sum(q_R * np.log(pi))


def _eval_E_lM(q_F, q_R, lM):
    """
    Evaluates :math:`\mathbb{E}[\log p(\tilde{b} | f, r; \theta)]`.

    Arguments
    ---------
    q_F : :class:`numpy.ndarray`, (C, 1, 3), 0 <= float <= 1
        Variational approximation of :math:`p(f | b, \tilde{b})`.
    q_R : :class:`numpy.ndarray`, (N, U, 2), 0 <= float <= 1
        Variational approximation of :math:`p(r | b, \tilde{b})`.
    lM : :class:`numpy.ndarray`, (C, U, 3, 3), float
        Mixture log densities of patients.
    """
    C = q_F.shape[0]
    (N, U) = q_R.shape[0:2]
    energy = 0
    for c in range(C):
        (n, m) = util.c_to_nm(c)
        q_R_w = _eval_q_R_w(q_R, n, m)
        for k in range(3):
            sum_ck = np.sum(q_R_w * lM[c, :, k, :])
            energy += q_F[c, 0, k] * sum_ck
    return energy


def _eval_E_lq_F(q_F, lq_F):
    """
    Evaluates the energy term related to :math:`\log q(f)`.

    Arguments
    ---------
    q_F : :class:`numpy.ndarray`, (C, 3), 0 <= float <= 1
        Variational approximation of :math:`p(f | b, \tilde{b})`.
    lq_F : :class:`numpy.ndarray`, (C, 3), 0 <= float <= 1
        Log variational approximation of :math:`p(f | b, \tilde{b})`.
    """
    return np.sum(q_F * lq_F)


def _eval_E_lq_R(q_R, lq_R):
    """
    Evaluates the energy term related to :math:`\log q(r)`.

    Arguments
    ---------
    q_R : :class:`numpy.ndarray`, (N, U, 2), 0 <= float <= 1
        Variational approximation of :math:`p(r | b, \tilde{b})`.
    lq_R : :class:`numpy.ndarray`, (N, U, 2), 0 <= float <= 1
        Log variational approximation of :math:`p(r | b, \tilde{b})`.
    """
    return np.sum(q_R * lq_R)


def _eval_dE_dm(q_F, q_R, dlN_dmj, dlM_dmj, j):
    """
    Evaluates :math:`\frac{\partial \mathcal{E}}{\partial \mu_j}`.

    Arguments
    ---------
    q_F : :class:`numpy.ndarray`, (C, 1, 3), 0 <= float <= 1
        Variational approximation of :math:`p(f | b, \tilde{b})`.
    q_R : :class:`numpy.ndarray`, (N, U, 2), 0 <= float <= 1
        Variational approximation of :math:`p(r | b, \tilde{b})`.
    dlN_dmj : :class:`numpy.ndarray`, (C, H), 0 <= float
        Derivative of log normal density w.r.t mu_j.
    dlM_dmj : :class:`numpy.ndarray`, (C, H, 3, 3), 0 <= float
        Derivative of log normal mixture w.r.t mu_j.
    """
    (C, H) = dlN_dmj.shape[0:2]
    (N, U) = q_R.shape[0:2]

    dE_dm = 0
    for c in range(C):
        (n, m) = util.c_to_nm(c)
        dE_dm -= q_F[c, 0, j] * np.sum(dlN_dmj[c, :])

        q_R_w = _eval_q_R_w(q_R, n, m)
        for k in range(3):
            dE_dm -= q_F[c, 0, k] * np.sum(q_R_w * dlM_dmj[c, :, k, :])

    return dE_dm


def _eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, k, l):
    """
    Computes the derivative of log M w.r.t. mu_j.

    Arguments
    ---------
    norm : :class:`numpy.ndarray`, (C, U, 3), 0 <= float
        Normal densities of patient correlations.
    mix : :class:`numpy.ndarray`, (C, U), 0 <= float
        Mixture densities of for k and l
    epsilon : 0 <= float <= 1
        Probability that a typical connection differs from the template.
    k : 0 <= k <= 2
        Index of connection type.

    Returns
    -------
    dlM_dm : :class:`numpy.ndarray`, (C, U,), 0 < float
        Derivative of mixture density w.r.t. mu.
    """

    eps = _eval_M_eps(eta, epsilon, l)
    if k != l:
        eps = 0.5 * (1 - eps)
    dlN_dm = _eval_dlN_dm(norm, mu, sigma)
    return eps * dlN_dm / mix


def _eval_dE_dh(q_R, q_F, norm, mix, epsilon):
    """
    Evaluates :math:`\frac{\partial \mathcal{E}}{\partial \eta}`.
    """
    (N, U) = q_R.shape[0:2]
    C = q_F.shape[0]

    dE_dh = 0
    for k in range(3):
        dlM_dh = _eval_dlM_dh(norm, mix[:, :, k, 2], epsilon, k)
        for c in range(C):
            (n, m) = util.c_to_nm(c)
            q_R_neq = q_R[n, :, 0] * q_R[m, :, 1]
            q_R_neq += q_R[n, :, 1] * q_R[m, :, 0]
            dE_dh -= q_F[c, 0, k] * np.sum(q_R_neq * dlM_dh[c, :])
    return dE_dh


def _eval_dlM_dh(norm, mix, epsilon, k):
    """
    Computes the derivative of log M w.r.t. eta.

    Arguments
    ---------
    norm : :class:`numpy.ndarray`, (C, U, 3), 0 <= float
        Normal densities of patient correlations.
    mix : :class:`numpy.ndarray`, (C, U), 0 <= float
        Mixture densities of for k and neq
    epsilon : 0 <= float <= 1
        Probability that a typical connection differs from the template.
    k : 0 <= k <= 2
        Index of connection type.

    Returns
    -------
    dlM_dh : :class:`numpy.ndarray`, (C, U,), 0 < float
        Derivative of mixture density w.r.t. eta.
    """
    eps = (2 * epsilon) - 1
    ls = list(set(range(3)) - set([k]))
    sum_norm_ls = np.sum(norm[:, :, ls], axis=2)
    return (eps * norm[:, :, k] - 0.5 * eps * sum_norm_ls) / mix


def _eval_dE_de(q_R, q_F, norm, mix, eta):
    """
    Evaluates :math:`\frac{\partial \mathcal{E}}{\partial \epsilon}`.
    """
    (N, U) = q_R.shape[0:2]
    C = q_F.shape[0]

    dE_de = 0
    for k in range(3):
        dlM_dh0 = _eval_dlM_de(norm, mix[:, :, k, 0], eta, k, 0)
        dlM_dh1 = _eval_dlM_de(norm, mix[:, :, k, 1], eta, k, 1)
        dlM_dh2 = _eval_dlM_de(norm, mix[:, :, k, 2], eta, k, 2)
        for c in range(C):
            (n, m) = util.c_to_nm(c)
            dlM_sum = q_R[n, :, 0] * q_R[m, :, 0] * dlM_dh0[c, :]
            dlM_sum += q_R[n, :, 1] * q_R[m, :, 1] * dlM_dh1[c, :]
            q_R_neq = q_R[n, :, 0] * q_R[m, :, 1]
            q_R_neq += q_R[n, :, 1] * q_R[m, :, 0]
            dlM_sum += q_R_neq * dlM_dh2[c, :]
            dE_de -= q_F[c, 0, k] * np.sum(dlM_sum)
    return dE_de


def _eval_dlM_de(norm, mix, eta, k, l):
    """
    Computes the derivative of log M w.r.t. epsilon.

    Arguments
    ---------
    norm : :class:`numpy.ndarray`, (C, U, 3), 0 <= float
        Normal densities of patient correlations.
    mix : :class:`numpy.ndarray`, (C, U), 0 <= float
        Mixture densities of for k and l
    eta : 0 <= float <= 1
        Probability of an anomalous connection btw a typical and anomalous region.
    k : 0 <= k <= 2
        Index of connection type.
    l : 0 <= k <= 2
        Index of mixture.

    Returns
    -------
    dlM_de : :class:`numpy.ndarray`, (C, U,), 0 < float
        Derivative of mixture density w.r.t. epsilon.
     """
    if l == 0:
        eps = -1
    elif l == 1:
        eps = 1
    else:
        eps = 2 * eta - 1
    ls = list(set(range(3)) - set([k]))
    sum_norm_ls = np.sum(norm[:, :, ls], axis=2)
    return (eps * norm[:, :, k] - 0.5 * eps * sum_norm_ls) / mix


def _eval_dlM_ds(bt, model, M_kl, N_j, j, k, l):
    """
    Computes the derivative of log M w.r.t. mu.
    """
    e = _eval_M_eps(eta, epsilon, l)
    if j != k:
        e = (1 - e) / 2
    return e * _eval_dN_ds(N, bt, mu, sigma) / M

def _eval_dN_dm(N, b, mu, sigma):
    """
    Computes the derivative of N w.r.t. mu.
    """
    return N * _eval_dlN_dm(b, mu, sigma)

def _eval_dlN_dm(b, mu, sigma):
    """
    Computes the derivative of log N w.r.t. mu.
    """
    return (b - mu) / (sigma * sigma)

def _eval_dN_ds(N, b, mu, sigma):
    """
    Computes the derivative of N w.r.t. sigma.
    """
    return N * _eval_dlN_ds(b, mu, sigma)

def _eval_dlN_ds(b, mu, sigma):
    """
    Computes the derivative of log N w.r.t. sigma.
    """
    diff = (b - mu)
    sigma2 = sigma * sigma
    return ((diff * diff) - sigma2) / (2 * sigma2)

