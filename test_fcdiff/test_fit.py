import numpy as np
import scipy.stats
import scipy.misc
import numpy.testing as nptest
import fcdiff
import unittest


# Utilities, not tests

def create_ideal_model():
    """
    Creates an ideal UnsharedRegionModel that should be easy to fit.
    """
    model = fcdiff.UnsharedRegionModel()
    model.pi = 0.1
    model.epsilon = 0.01
    model.eta = 0.3
    model.gamma = np.ones((3,)) / 3
    model.mu = np.array([-0.5, 0, 0.5])
    model.sigma = np.ones((3,)) * 0.05
    return model


def rand(lower, upper, shape, seed = 0):
    """
    Generates random numbers in (lower, upper].

    Arguments
    ---------
    lower : float
        Lower open boundary of interval.
    upper : float
        Upper closed boundary of interval.
    shape : list<int>
        Shape of array to create
    seed : uint
        Seed for RNG creation.

    Returns
    -------
    Array of random numbers.
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(lower, upper, size = shape)


def rand_prob(shape, seed = 0):
    """
    Generates random probabilities from uniform in [1e-7, 1).
    """
    return rand(1e-7, 1, shape, seed = seed)


def rand_prob_vector(shape, seed = 0):
    """
    Generates random probability vectors from uniform in [1e-7, 1).

    The last dimension should sum to 1.
    """
    prob = rand_prob(shape, seed = seed)
    prob_sum = np.sum(prob, axis = -1, keepdims = True)
    prob /= prob_sum
    return prob


# Class tests

class UnsharedRegionFitTest(unittest.TestCase):

    def test_init_lps_shape(self):
        """ Tests shape of log prob distributions after init.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        (N, C, H, U) = (4, 6, 7, 5)
        fit._init_lps(N, H, U)
        nptest.assert_equal(fit._lq_R.shape, (N, U, 2))
        nptest.assert_allclose(np.sum(np.exp(fit._lq_R), axis = 2), 1)
        nptest.assert_equal(fit._lq_F.shape, (C, 1, 3))
        nptest.assert_allclose(np.sum(np.exp(fit._lq_F), axis = 2), 1)
        nptest.assert_equal(fit._lp_B_g_F.shape, (C, H, 3))
        nptest.assert_equal(fit._p_Bt_g_Ft.shape, (C, U, 3))
        nptest.assert_equal(fit._lM.shape, (C, U, 3, 3))

    def test_is_converged_increase(self):
        """ Tests convergence when there is an energy increase.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.rel_tol = 0.5
        fit.energy = [1, 1.25]
        nptest.assert_equal(fit._is_converged(1), True)

    def test_is_converged_same(self):
        """ Tests convergence when the energy does not change.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.rel_tol = 0.5
        fit.energy = [1, 1]
        nptest.assert_equal(fit._is_converged(1), True)

    def test_is_converged_decrease_small(self):
        """ Tests convergence when energy decrease is small enough.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.rel_tol = 0.5
        fit.energy = [1, 0.501]
        nptest.assert_equal(fit._is_converged(1), True)

    def test_is_converged_decrease_medium(self):
        """ Tests lack of convergence when energy decrease is just big enough.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.rel_tol = 0.5
        fit.energy = [1, 0.5]
        nptest.assert_equal(fit._is_converged(1), False)

    def test_is_converged_decrease_big(self):
        """ Tests lack of convergence when energy decrease is big enough.
        """
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.rel_tol = 0.5
        fit.energy = [1, 0.499]
        nptest.assert_equal(fit._is_converged(1), False)

    def test_update_lps(self):
        """ Tests updating log probabilities.
        """
        (N, C, H, U) = (4, 6, 7, 5)
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.b = 1 - 2 * rand_prob((C, H), seed = 0)
        fit.bt = 1 - 2 * rand_prob((C, U), seed = 1)
        fit.model = create_ideal_model()
        fit._init_lps(N, H, U)
        fit._update_lps()

        lp_B_g_F = np.zeros((C, H, 3))
        p_Bt_g_Ft = np.zeros((C, U, 3))
        lM = np.zeros((C, U, 3, 3))
        for c in range(C):
            for k in range(3):
                mu = fit.model.mu[k]
                sigma = fit.model.sigma[k]
                for h in range(H):
                    lp_B_g_F[c, h, k] = scipy.stats.norm.logpdf(fit.b[c, h], mu, sigma)
                for u in range(U):
                    p_Bt_g_Ft[c, u, k] = scipy.stats.norm.pdf(fit.bt[c, u], mu, sigma)
        eta = fit.model.eta
        epsilon = fit.model.epsilon
        for c in range(C):
            for u in range(U):
                for k in range(3):
                    for l in range(3):
                        N = p_Bt_g_Ft[c, u, :].reshape((1, 1, 3))
                        M = fcdiff.fit._eval_M(N, eta, epsilon, k, l)
                        lM[c, u, k, l] = np.log(M)

        nptest.assert_equal(fit._lp_B_g_F, lp_B_g_F)
        nptest.assert_equal(fit._p_Bt_g_Ft, p_Bt_g_Ft)
        nptest.assert_equal(fit._lM, lM)

    def test_opt_jac_eta(self):
        """ Tests computation of jacobian of minimization criterion wrt eta.
        """
        (N, C, H, U) = (4, 6, 7, 5)
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.b = rand_prob_vector((C, H))
        fit.bt = rand_prob_vector((C, U))
        fit.model = create_ideal_model();
        fit._init_lps(N, H, U)
        fit._update_lps()

        jac = fit._opt_jac()
        jac_eta = jac[0]

        exp_jac_eta = 0
        M = np.exp(fit._lM)
        q_R = np.exp(fit._lq_R)
        q_F = np.exp(fit._lq_F)
        for c in range(C):
            (n, m) = fcdiff.util.c_to_nm(c)
            for k in range(3):
                ls = list(set(range(3)) - set([k]))
                sum_u = 0
                for u in range(U):
                    sum_p_Bt_g_Ft_ls = np.sum(fit._p_Bt_g_Ft[c, u, ls])
                    w = (2 * fit.model.epsilon) - 1
                    dM_dh = w * fit._p_Bt_g_Ft[c, u, k] - 0.5 * w * sum_p_Bt_g_Ft_ls
                    dlM_dh = dM_dh / M[c, u, k, 2]
                    q_R_cu = q_R[n, u, 0] * q_R[m, u, 1]
                    q_R_cu += q_R[n, u, 1] * q_R[m, u, 0]
                    sum_u += q_R_cu * dlM_dh

                exp_jac_eta -= q_F[c, 0, k] * sum_u

        nptest.assert_allclose(jac_eta, exp_jac_eta)

    def test_opt_jac_epsilon(self):
        """ Tests computation of jacobian of minimization criterion wrt epsilon.
        """
        (N, C, H, U) = (4, 6, 7, 5)
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.b = rand_prob_vector((C, H))
        fit.bt = rand_prob_vector((C, U))
        fit.model = create_ideal_model();
        fit._init_lps(N, H, U)
        fit._update_lps()

        jac = fit._opt_jac()
        jac_eps = jac[1]

        exp_jac_eps = 0
        M = np.exp(fit._lM)
        q_R = np.exp(fit._lq_R)
        q_F = np.exp(fit._lq_F)
        for c in range(C):
            (n, m) = fcdiff.util.c_to_nm(c)
            for k in range(3):
                ls = list(set(range(3)) - set([k]))
                sum_u = 0
                for u in range(U):
                    sum_p_Bt_g_Ft_ls = np.sum(fit._p_Bt_g_Ft[c, u, ls])

                    dM0_dh = 0.5 * sum_p_Bt_g_Ft_ls - fit._p_Bt_g_Ft[c, u, k]
                    dlM0_dh = dM0_dh / M[c, u, k, 0]
                    q_R_00 = q_R[n, u, 0] * q_R[m, u, 0]
                    sum_u += q_R_00 * dlM0_dh

                    dM1_dh = fit._p_Bt_g_Ft[c, u, k] - 0.5 * sum_p_Bt_g_Ft_ls
                    dlM1_dh = dM1_dh / M[c, u, k, 1]
                    q_R_11 = q_R[n, u, 1] * q_R[m, u, 1]
                    sum_u += q_R_11 * dlM1_dh

                    w = (2 * fit.model.eta - 1)
                    dMneq_dh = w * fit._p_Bt_g_Ft[c, u, k] - 0.5 * w * sum_p_Bt_g_Ft_ls
                    dlMneq_dh = dMneq_dh / M[c, u, k, 2]
                    q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                    q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                    sum_u += q_R_neq * dlMneq_dh

                exp_jac_eps -= q_F[c, 0, k] * sum_u

        nptest.assert_allclose(jac_eps, exp_jac_eps)

    def test_opt_jac_mu(self):
        """ Tests computation of jacobian of minimization criterion wrt mu.
        """
        (N, C, H, U) = (4, 6, 7, 5)
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.b = rand_prob_vector((C, H))
        fit.bt = rand_prob_vector((C, U))
        fit.model = create_ideal_model();
        fit._init_lps(N, H, U)
        fit._update_lps()

        epsilon_t = fit.model.eta * fit.model.epsilon + (1 - fit.model.eta) * (1 - fit.model.epsilon)

        jac = fit._opt_jac()
        jac_mu = jac[2:5]

        exp_jac_mu = np.zeros((3,))
        M = np.exp(fit._lM)
        q_R = np.exp(fit._lq_R)
        q_F = np.exp(fit._lq_F)
        for j in range(3):
            for c in range(C):
                (n, m) = fcdiff.util.c_to_nm(c)

                sum_h = 0
                for h in range(H):
                    dlNj_dm = (fit.b[c, h] - fit.model.mu[j]) / (fit.model.sigma[j]**2)
                    sum_h += dlNj_dm
                exp_jac_mu[j] -= q_F[c, 0, j] * sum_h

                for k in range(3):
                    sum_u = 0
                    for u in range(U):
                        dlNj_dm = (fit.bt[c, u] - fit.model.mu[j]) / (fit.model.sigma[j]**2)
                        dNj_dm = fit._p_Bt_g_Ft[c, u, j] * dlNj_dm

                        if j == k:
                            w0 = 1 - fit.model.epsilon
                        else:
                            w0 = fit.model.epsilon / 2
                        dM0_dm = w0 * dNj_dm
                        dlM0_dm = dM0_dm / M[c, u, k, 0]
                        q_R_00 = q_R[n, u, 0] * q_R[m, u, 0]
                        sum_u += q_R_00 * dlM0_dm

                        if j == k:
                            w1 = fit.model.epsilon
                        else:
                            w1 = (1 - fit.model.epsilon) / 2
                        dM1_dm = w1 * dNj_dm
                        dlM1_dm = dM1_dm / M[c, u, k, 1]
                        q_R_11 = q_R[n, u, 1] * q_R[m, u, 1]
                        sum_u += q_R_11 * dlM1_dm

                        if j == k:
                            wneq = epsilon_t
                        else:
                            wneq = (1 - epsilon_t) / 2
                        dMneq_dm = wneq * dNj_dm
                        dlMneq_dm = dMneq_dm / M[c, u, k, 2]
                        q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                        q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                        sum_u += q_R_neq * dlMneq_dm
                    exp_jac_mu[j] -= q_F[c, 0, k] * sum_u

        nptest.assert_allclose(jac_mu, exp_jac_mu)

    def test_opt_jac_sigma(self):
        """ Tests computation of jacobian of minimization criterion wrt sigma.
        """
        (N, C, H, U) = (4, 6, 7, 5)
        fit = fcdiff.fit.UnsharedRegionFit()
        fit.b = rand_prob_vector((C, H))
        fit.bt = rand_prob_vector((C, U))
        fit.model = create_ideal_model();
        fit._init_lps(N, H, U)
        fit._update_lps()

        epsilon_t = fit.model.eta * fit.model.epsilon + (1 - fit.model.eta) * (1 - fit.model.epsilon)

        jac = fit._opt_jac()
        jac_sigma = jac[5:8]

        exp_jac_sigma = np.zeros((3,))
        M = np.exp(fit._lM)
        q_R = np.exp(fit._lq_R)
        q_F = np.exp(fit._lq_F)
        sigma2 = fit.model.sigma ** 2
        for j in range(3):
            for c in range(C):
                (n, m) = fcdiff.util.c_to_nm(c)

                sum_h = 0
                for h in range(H):
                    diff = fit.b[c, h] - fit.model.mu[j]
                    dlNj_ds = (diff ** 2 - sigma2[j]) / (2 * sigma2[j])
                    sum_h += dlNj_ds
                exp_jac_sigma[j] -= q_F[c, 0, j] * sum_h

                for k in range(3):
                    sum_u = 0
                    for u in range(U):
                        diff = fit.bt[c, u] - fit.model.mu[j]
                        dlNj_ds = (diff**2 - sigma2[j]) / (2 * sigma2[j])
                        dNj_ds = fit._p_Bt_g_Ft[c, u, j] * dlNj_ds

                        if j == k:
                            w0 = 1 - fit.model.epsilon
                        else:
                            w0 = fit.model.epsilon / 2
                        dM0_ds = w0 * dNj_ds
                        dlM0_ds = dM0_ds / M[c, u, k, 0]
                        q_R_00 = q_R[n, u, 0] * q_R[m, u, 0]
                        sum_u += q_R_00 * dlM0_ds

                        if j == k:
                            w1 = fit.model.epsilon
                        else:
                            w1 = (1 - fit.model.epsilon) / 2
                        dM1_ds = w1 * dNj_ds
                        dlM1_ds = dM1_ds / M[c, u, k, 1]
                        q_R_11 = q_R[n, u, 1] * q_R[m, u, 1]
                        sum_u += q_R_11 * dlM1_ds

                        if j == k:
                            wneq = epsilon_t
                        else:
                            wneq = (1 - epsilon_t) / 2
                        dMneq_ds = wneq * dNj_ds
                        dlMneq_ds = dMneq_ds / M[c, u, k, 2]
                        q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                        q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                        sum_u += q_R_neq * dlMneq_ds
                    exp_jac_sigma[j] -= q_F[c, 0, k] * sum_u

        nptest.assert_allclose(jac_sigma, exp_jac_sigma)

    def test_opt_fun():
        """ Tests evaluation the optmization function and jacobian.
        """

        # create an ideal model and sample from it
        (N, C, H, U) = (4, 6, 7, 5)
        model = create_ideal_model()
        (r, t, f, ft, b, bt) = model.sample(N, H, U)

        # give almost perfect estimates of log posteriors
        r = r.clip(1e-5, 1 - 1e-5)
        q_R = np.zeros((N, U, 2))
        q_R[:, :, 0] = 1 - r
        q_R[:, :, 1] = r
        lq_R = np.log(q_R)

        f = f.clip(1e-5, 1 - 1e-5)
        q_F = f.reshape((C, 1, 3))
        lq_F = np.log(q_F)

        theta_sub = fcdiff.fit._pack_theta_sub(model)
        lp_B_g_F = np.zeros((C, H, 3))
        p_Bt_g_Ft = np.zeros((C, U, 3))
        lM = np.zeros((C, U, 3, 3))

        # evaluate the ideal model's energy
        (ideal_energy, ideal_jac) = fcdiff.fit._opt_fun(theta_sub, q_F, q_R,
                lq_F, lq_R, lp_B_g_F, p_Bt_g_Ft, lM, model, b, bt)

        # perturb the parameter values a little and re-evaluate the energy
        model.eta += 0.1
        model.epsilon += 0.01
        model.mu += [-0.1, 0, 0.1]
        model.sigma += [0.01, 0, 0.01]

        (pert_energy, pert_jac) = fcdiff.fit._opt_fun(theta_sub, q_F, q_R,
                lq_F, lq_R, lp_B_g_F, p_Bt_g_Ft, lM, model, b, bt)

        # check that the energy went up
        nptest.assert_array_less(ideal_energy, pert_energy)

#def test_update_theta_sub():
#    """ Tests updating the vector of parameters excluding pi and gamma.
#    """
#
#    # create an ideal model and sample from it
#    (N, H, U) = (60, 50, 40)
#    C = fcdiff.N_to_C(N)
#    model = create_ideal_model()
#    (r, t, f, ft, b, bt) = model.sample(N, H, U)
#
#    # give almost perfect estimates of log posteriors
#    r = r.clip(1e-5, 1 - 1e-5)
#    q_R = np.zeros((N, U, 2))
#    q_R[:, :, 0] = 1 - r
#    q_R[:, :, 1] = r
#    f = f.clip(1e-5, 1 - 1e-5)
#    q_F = np.reshape(f, (C, 1, 3))
#
#    # perturb the parameter values a little
#    model.eta += 0.1
#    #model.epsilon += 0.01
#    #model.mu += [-0.05, 0, 0.05]
#    #model.sigma += [0.01, 0.01, 0.01]
#
#    # initialize (log) probabilites
#    lp_B_g_F = np.zeros((C, H, 3))
#    p_Bt_g_Ft = np.zeros((C, U, 3))
#    lM = np.zeros((C, U, 3, 3))
#
#    fcdiff.fit._update_theta_sub(model, b, bt, q_R, q_F, lp_B_g_F, p_Bt_g_Ft, lM)
#
#    exp_model = create_ideal_model()
#
#    nptest.assert_allclose(model.eta, exp_model.eta)
#    nptest.assert_allclose(model.epsilon, exp_model.epsilon)
#    nptest.assert_allclose(model.mu, exp_model.mu)
#    nptest.assert_allclose(model.sigma, exp_model.sigma)


# Function tests
def test_eval_E_lp_F():
    """
    Tests evaluation of E[log p(f)].
    """
    (C, K) = (2, 3)
    q_F = rand_prob_vector((C, 1, K))
    model = create_ideal_model()
    act_E = fcdiff.fit._eval_E_lp_F(q_F, model.gamma)
    exp_E = 0
    for c in range(C):
        for k in range(K):
            exp_E += q_F[c, 0, k] * np.log(model.gamma[k])
    nptest.assert_allclose(act_E, exp_E)


def test_eval_E_lp_B_g_F():
    """
    Tests evaluation of E[log p(b | f)].
    """
    (C, H, K) = (2, 4, 3)
    lp_B_g_F = np.log(rand_prob((C, H, K)))
    q_F = rand_prob((C, 1, K))
    act_E = fcdiff.fit._eval_E_lp_B_g_F(q_F, lp_B_g_F)
    exp_E = 0
    for c in range(C):
        for h in range(H):
            for k in range(K):
                exp_E += q_F[c, 0, k] * lp_B_g_F[c, h, k]
    nptest.assert_allclose(act_E, exp_E)


def test_eval_E_lp_R():
    """
    Tests evaluation of E[log p(r)].
    """
    (N, U) = (3, 4)
    model = create_ideal_model()
    q_R = rand_prob_vector((N, U, 2))
    pi = [1 - model.pi, model.pi]
    act_E = fcdiff.fit._eval_E_lp_R(q_R, pi)
    exp_E = 0
    for n in range(N):
        for u in range(U):
            exp_E += q_R[n, u, 0] * np.log(pi[0])
            exp_E += q_R[n, u, 1] * np.log(pi[1])
    nptest.assert_allclose(act_E, exp_E)


def test_eval_E_lq_F():
    """
    Tests evaluation of E[log q(f)].
    """
    (C, K) = (2, 3)
    q_F = rand_prob_vector((C, 1, K))
    lq_F = np.log(q_F)
    act_E = fcdiff.fit._eval_E_lq_F(q_F, lq_F)
    exp_E = 0
    for c in range(C):
        for k in range(K):
            exp_E += q_F[c, 0, k] * lq_F[c, 0, k]
    nptest.assert_allclose(act_E, exp_E)


def test_eval_M_00():
    """
    Tests evaluation of mixture density M_00.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 0, 0)

    eps = 1 - model.epsilon
    exp_M = 0
    exp_M += eps * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_01():
    """
    Tests evaluation of mixture density M_01.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 0, 1)

    eps = model.epsilon
    exp_M = 0
    exp_M += eps * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_02():
    """
    Tests evaluation of mixture density M_02.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 0, 2)

    eps = model.eta * model.epsilon
    eps += (1 - model.eta) * (1 - model.epsilon)
    exp_M = 0
    exp_M += eps * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_10():
    """
    Tests evaluation of mixture density M_10.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 1, 0)

    eps = 1 - model.epsilon
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += eps * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_11():
    """
    Tests evaluation of mixture density M_11.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 1, 1)

    eps = model.epsilon
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += eps * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_12():
    """
    Tests evaluation of mixture density M_12.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 1, 2)

    eps = model.eta * model.epsilon
    eps += (1 - model.eta) * (1 - model.epsilon)
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += eps * p_Bt_g_Ft[:, :, 1]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_20():
    """
    Tests evaluation of mixture density M_20.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 2, 0)

    eps = 1 - model.epsilon
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += eps * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_21():
    """
    Tests evaluation of mixture density M_21.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 2, 1)

    eps = model.epsilon
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += eps * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_M_22():
    """
    Tests evaluation of mixture density M_22.
    """
    model = create_ideal_model()
    p_Bt_g_Ft = rand_prob_vector((1, 1, 3))
    act_M = fcdiff.fit._eval_M(p_Bt_g_Ft, model.eta, model.epsilon, 2, 2)

    eps = model.eta * model.epsilon
    eps += (1 - model.eta) * (1 - model.epsilon)
    exp_M = 0
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 0]
    exp_M += 0.5 * (1 - eps) * p_Bt_g_Ft[:, :, 1]
    exp_M += eps * p_Bt_g_Ft[:, :, 2]

    nptest.assert_allclose(act_M, exp_M)


def test_eval_E_lM():
    """
    Tests computation of E[log p(bt | f, r)]
    """
    (N, U, K) = (2, 4, 3)
    C = fcdiff.N_to_C(N)
    q_F = rand_prob_vector((C, 1, K))
    q_R = rand_prob_vector((N, U, K))
    lM = np.log(rand_prob_vector((C, U, K, 3)))
    E = fcdiff.fit._eval_E_lM(q_F, q_R, lM)
    exp_E = 0
    for c in range(C):
        (n, m) = fcdiff.c_to_nm(c)
        for u in range(U):
            for k in range(K):
                exp_E += q_F[c, 0, k] * q_R[n, u, 0] * q_R[m, u, 0] * lM[c, u, k, 0]
                exp_E += q_F[c, 0, k] * q_R[n, u, 1] * q_R[m, u, 1] * lM[c, u, k, 1]
                q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                exp_E += q_F[c, 0, k] * q_R_neq * lM[c, u, k, 2]
    nptest.assert_allclose(E, exp_E)


def test_eval_E_lq_R():
    """
    Tests computation of E[log q(r)].
    """
    (N, U, K) = (3, 4, 2)
    q_R = rand_prob_vector((N, U, K))
    lq_R = np.log(q_R)
    E = fcdiff.fit._eval_E_lq_R(q_R, lq_R)
    exp_E = 0
    for n in range(N):
        for u in range(U):
            for k in range(K):
                exp_E += q_R[n, u, k] * lq_R[n, u, k]
    nptest.assert_allclose(E, exp_E)


def test_update_lq_F():
    """
    Tests update of log q(f).
    """
    (N, H, U) = (6, 5, 4)
    C = fcdiff.N_to_C(N)

    q_R = rand_prob_vector((N, U, 2))
    lp_B_g_F = np.log(rand_prob((C, H, 3)))
    lM = np.log(rand_prob((C, U, 3, 3)))

    fit = fcdiff.fit.UnsharedRegionFit()
    fit._lq_R = np.log(q_R)
    fit._lp_B_g_F = lp_B_g_F
    fit._lM = lM
    fit.model = fcdiff.model.UnsharedRegionModel()
    fit.model.gamma = rand_prob_vector((3,))

    fit._update_lq_F()
    act_lq_F = fit._lq_F

    exp_lq_F = np.zeros((C, 1, 3))
    for k in range(3):
        exp_lq_F[:, :, k] += np.log(fit.model.gamma[k])
    for c in range(C):
        (n, m) = fcdiff.util.c_to_nm(c)
        for k in range(3):
            for h in range(H):
                exp_lq_F[c, :, k] += lp_B_g_F[c, h, k]
            for u in range(U):
                q_R_0 = q_R[n, u, 0] * q_R[m, u, 0]
                q_R_1 = q_R[n, u, 1] * q_R[m, u, 1]
                q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                exp_lq_F[c, :, k] += q_R_0 * lM[c, u, k, 0]
                exp_lq_F[c, :, k] += q_R_1 * lM[c, u, k, 1]
                exp_lq_F[c, :, k] += q_R_neq * lM[c, u, k, 2]
    exp_lq_F -= scipy.misc.logsumexp(exp_lq_F, axis = 2, keepdims = True)

    nptest.assert_allclose(act_lq_F, exp_lq_F)


def test_update_lq_R():
    """
    Tests update of log q(r).
    """
    (N, U) = (6, 4)
    C = fcdiff.N_to_C(N)

    pi = rand_prob_vector((2,))
    q_R = rand_prob_vector((N, U, 2))
    q_F = rand_prob_vector((C, 1, 3))
    lM = np.log(rand_prob((C, U, 3, 3)))

    fit = fcdiff.fit.UnsharedRegionFit()
    fit._lq_R = np.log(q_R)
    fit._lq_F = np.log(q_F)
    fit._lM = lM
    fit.model = fcdiff.model.UnsharedRegionModel()
    fit.model.pi = pi

    fit._update_lq_R()
    act_lq_R = fit._lq_R

    exp_lq_R = np.zeros((N, U, 2))
    for n in range(N):
        for u in range(U):
            exp_lq_R[n, u, 0] = np.log(pi[0])
            exp_lq_R[n, u, 1] = np.log(pi[1])
            for m in list(set(range(N)) - set([n])):
                c = fcdiff.util.nm_to_c(n, m)
                for k in range(3):
                    lM_00 = q_R[m, u, 0] * lM[c, u, k, 0]
                    lM_1neq = q_R[m, u, 1] * lM[c, u, k, 2]
                    exp_lq_R[n, u, 0] += q_F[c, :, k] * (lM_00 + lM_1neq)

                    lM_11 = q_R[m, u, 1] * lM[c, u, k, 1]
                    lM_0neq = q_R[m, u, 0] * lM[c, u, k, 2]
                    exp_lq_R[n, u, 1] += q_F[c, :, k] * (lM_11 + lM_0neq)
            exp_lq_R[n, u, :] -= scipy.misc.logsumexp(exp_lq_R[n, u, :], keepdims = True)
            q_R[n, u, :] = np.exp(exp_lq_R[n, u, :])

    nptest.assert_allclose(act_lq_R, exp_lq_R)


def test_update_pi():
    """
    Tests update of parameter pi.
    """
    (N, U) = (6, 4)

    q_R = rand_prob_vector((N, U, 2))
    fit = fcdiff.fit.UnsharedRegionFit()
    fit.model = fcdiff.model.UnsharedRegionModel()
    fit._lq_R = np.log(q_R)

    fit._update_pi()
    act_pi = fit.model.pi

    exp_pi = 0
    for n in range(N):
        for u in range(U):
            exp_pi += q_R[n, u, 1]
    exp_pi /= (N * U)

    nptest.assert_allclose(act_pi, exp_pi)


def test_update_gamma():
    """
    Tests update of parameter gamma.
    """
    (N, U) = (6, 4)
    C = fcdiff.N_to_C(N)

    q_F = rand_prob_vector((C, 1, 3))
    fit = fcdiff.fit.UnsharedRegionFit()
    fit.model = fcdiff.model.UnsharedRegionModel()
    fit._lq_F = np.log(q_F)

    fit._update_gamma()
    act_gamma = fit.model.gamma

    exp_gamma = np.zeros((3,))
    for c in range(C):
        for k in range(3):
            exp_gamma[k] += q_F[c, 0, k]
    exp_gamma /= C

    nptest.assert_allclose(act_gamma, exp_gamma)


def test_eval_dE_dm0():
    """
    Tests evaluation of derivative of the energy w.r.t. mu_0.
    """
    (N, H, U) = (6, 4, 5)
    C = fcdiff.util.N_to_C(N)

    dlN_dm = rand(-10, 10, (C, H))
    dlM_dm = rand(-10, 10, (C, U, 3, 3))
    q_F = rand_prob_vector((C, 1, 3))
    q_R = rand_prob_vector((N, U, 2))

    act_dE_dm = fcdiff.fit._eval_dE_dm(q_F, q_R, dlN_dm, dlM_dm, 0)

    exp_dE_dm = 0
    for c in range(C):
        (n, m) = fcdiff.util.c_to_nm(c)
        sum_h = 0
        for h in range(H):
            sum_h += dlN_dm[c, h]
        exp_dE_dm -= q_F[c, 0, 0] * sum_h

        for k in range(3):
            sum_u = 0
            for u in range(U):
                q_R_00 = q_R[n, u, 0] * q_R[m, u, 0]
                q_R_11 = q_R[n, u, 1] * q_R[m, u, 1]
                q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]

                sum_u += q_R_00 * dlM_dm[c, u, k, 0]
                sum_u += q_R_11 * dlM_dm[c, u, k, 1]
                sum_u += q_R_neq * dlM_dm[c, u, k, 2]
            exp_dE_dm -= q_F[c, 0, k] * sum_u

    nptest.assert_allclose(act_dE_dm, exp_dE_dm)


def test_eval_dlM_dm_00():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 0, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 0, 0)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = (1 - epsilon) * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_01():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 0, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 0, 1)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = 0.5 * (1 - epsilon) * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_02():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 0, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 0, 2)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    eps = eta * epsilon + (1 - eta) * (1 - epsilon)
    exp_dlM_dm = 0.5 * (1 - eps) * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_10():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 1, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 1, 0)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = 0.5 * epsilon * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_11():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 1, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 1, 1)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = epsilon * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_12():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 1, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 1, 2)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    eps = eta * epsilon + (1 - eta) * (1 - epsilon)
    exp_dlM_dm = 0.5 * (1 - eps) * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_20():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 2, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 2, 0)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = 0.5 * epsilon * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_21():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 2, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 2, 1)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    exp_dlM_dm = 0.5 * (1 - epsilon) * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dm_22():
    """
    Tests evaluation of derivative of log M w.r.t. mu for k = 2, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U))
    mix = rand_prob((C, U))
    mu = 0.14
    sigma = 0.02
    epsilon = 0.07
    eta = 0.29

    act_dlM_dm = fcdiff.fit._eval_dlM_dm(norm, mix, mu, sigma, eta, epsilon, 2, 2)

    dlN_dm = fcdiff.fit._eval_dlN_dm(norm, mu, sigma)
    eps = eta * epsilon + (1 - eta) * (1 - epsilon)
    exp_dlM_dm = eps * dlN_dm / mix

    nptest.assert_allclose(act_dlM_dm, exp_dlM_dm)


def test_eval_dlM_dh_0():
    """
    Tests evaluation of derivative of log M w.r.t. eta for k = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    epsilon = 0.14

    act_dlM_dh = fcdiff.fit._eval_dlM_dh(norm, mix, epsilon, 0)

    sum_norm_ls = norm[:, :, 1] + norm[:, :, 2]
    exp_dlM_dh = (2 * epsilon - 1) * norm[:, :, 0]
    exp_dlM_dh -= 0.5 * (2 * epsilon - 1) * sum_norm_ls
    exp_dlM_dh /= mix

    nptest.assert_allclose(act_dlM_dh, exp_dlM_dh)


def test_eval_dlM_dh_1():
    """
    Tests evaluation of derivative of log M w.r.t. eta for k = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    epsilon = 0.14

    act_dlM_dh = fcdiff.fit._eval_dlM_dh(norm, mix, epsilon, 1)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 2]
    exp_dlM_dh = (2 * epsilon - 1) * norm[:, :, 1]
    exp_dlM_dh -= 0.5 * (2 * epsilon - 1) * sum_norm_ls
    exp_dlM_dh /= mix

    nptest.assert_allclose(act_dlM_dh, exp_dlM_dh)


def test_eval_dlM_dh_2():
    """
    Tests evaluation of derivative of log M w.r.t. eta for k = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    epsilon = 0.14

    act_dlM_dh = fcdiff.fit._eval_dlM_dh(norm, mix, epsilon, 2)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 1]
    exp_dlM_dh = (2 * epsilon - 1) * norm[:, :, 2]
    exp_dlM_dh -= 0.5 * (2 * epsilon - 1) * sum_norm_ls
    exp_dlM_dh /= mix

    nptest.assert_allclose(act_dlM_dh, exp_dlM_dh)


def test_eval_dlM_de_00():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 0, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 0, 0)

    sum_norm_ls = norm[:, :, 1] + norm[:, :, 2]
    exp_dlM_de = 0.5 * sum_norm_ls - norm[:, :, 0]
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_10():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 1, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 1, 0)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 2]
    exp_dlM_de = 0.5 * sum_norm_ls - norm[:, :, 1]
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_20():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 2, l = 0.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 2, 0)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 1]
    exp_dlM_de = 0.5 * sum_norm_ls - norm[:, :, 2]
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_01():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 0, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 0, 1)

    sum_norm_ls = norm[:, :, 1] + norm[:, :, 2]
    exp_dlM_de = norm[:, :, 0] - 0.5 * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_11():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 1, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 1, 1)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 2]
    exp_dlM_de = norm[:, :, 1] - 0.5 * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_21():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 2, l = 1.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 2, 1)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 1]
    exp_dlM_de = norm[:, :, 2] - 0.5 * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_02():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 0, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 0, 2)

    sum_norm_ls = norm[:, :, 1] + norm[:, :, 2]
    exp_dlM_de = (2 * eta - 1) * norm[:, :, 0]
    exp_dlM_de -= 0.5 * (2 * eta - 1) * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_12():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 1, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 1, 2)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 2]
    exp_dlM_de = (2 * eta - 1) * norm[:, :, 1]
    exp_dlM_de -= 0.5 * (2 * eta - 1) * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dlM_de_22():
    """
    Tests evaluation of derivative of log M wrt epsilon for k = 2, l = 2.
    """
    (N, U) = (6, 5)
    C = fcdiff.util.N_to_C(N)
    norm = rand_prob((C, U, 3))
    mix = rand_prob((C, U))
    eta = 0.33

    act_dlM_de = fcdiff.fit._eval_dlM_de(norm, mix, eta, 2, 2)

    sum_norm_ls = norm[:, :, 0] + norm[:, :, 1]
    exp_dlM_de = (2 * eta - 1) * norm[:, :, 2]
    exp_dlM_de -= 0.5 * (2 * eta - 1) * sum_norm_ls
    exp_dlM_de /= mix

    nptest.assert_allclose(act_dlM_de, exp_dlM_de)


def test_eval_dE_dh():
    """
    Tests evaluation :math:`\frac{\partial \mathcal{E}}{\partial \eta}`.
    """
    (N, U) = (6, 4)
    C = fcdiff.N_to_C(N)

    q_R = rand_prob_vector((N, U, 2))
    q_F = rand_prob_vector((C, 1, 3))
    mix = rand_prob((C, U, 3, 3))
    norm = rand_prob((C, U, 3))
    epsilon = 0.08

    act_dE_dh = fcdiff.fit._eval_dE_dh(q_R, q_F, norm, mix, epsilon)

    exp_dE_dh = 0
    for k in range(3):
        dlM_dh = fcdiff.fit._eval_dlM_dh(norm, mix[:, :, k, 2], epsilon, k)
        for c in range(C):
            (n, m) = fcdiff.util.c_to_nm(c)
            u_sum = 0
            for u in range(U):
                q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                u_sum += q_R_neq * dlM_dh[c, u]
            exp_dE_dh -= q_F[c, 0, k] * u_sum

    nptest.assert_allclose(act_dE_dh, exp_dE_dh)


def test_eval_dE_de():
    """
    Tests evaluation :math:`\frac{\partial \mathcal{E}}{\partial \epsilon}`.
    """
    (N, U) = (6, 4)
    C = fcdiff.N_to_C(N)

    q_R = rand_prob_vector((N, U, 2))
    q_F = rand_prob_vector((C, 1, 3))
    mix = rand_prob((C, U, 3, 3))
    norm = rand_prob((C, U, 3))
    eta = 0.33

    act_dE_de = fcdiff.fit._eval_dE_de(q_R, q_F, norm, mix, eta)

    exp_dE_de = 0
    for k in range(3):
        dlM_dh0 = fcdiff.fit._eval_dlM_de(norm, mix[:, :, k, 0], eta, k, 0)
        dlM_dh1 = fcdiff.fit._eval_dlM_de(norm, mix[:, :, k, 1], eta, k, 1)
        dlM_dh2 = fcdiff.fit._eval_dlM_de(norm, mix[:, :, k, 2], eta, k, 2)
        for c in range(C):
            (n, m) = fcdiff.util.c_to_nm(c)
            u_sum = 0
            for u in range(U):
                q_R_0 = q_R[n, u, 0] * q_R[m, u, 0]
                q_R_1 = q_R[n, u, 1] * q_R[m, u, 1]
                q_R_neq = q_R[n, u, 0] * q_R[m, u, 1]
                q_R_neq += q_R[n, u, 1] * q_R[m, u, 0]
                u_sum += q_R_0 * dlM_dh0[c, u]
                u_sum += q_R_1 * dlM_dh1[c, u]
                u_sum += q_R_neq * dlM_dh2[c, u]
            exp_dE_de -= q_F[c, 0, k] * u_sum

    nptest.assert_allclose(act_dE_de, exp_dE_de)


