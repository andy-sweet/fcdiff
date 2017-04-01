import sys, os, unittest
import numpy as np
import numpy.testing as nptest
import fcdiff

class UnsharedRegionModelTest(unittest.TestCase):

    def test_str(self):
        """ Tests informal string representation.
        """
        model = fcdiff.UnsharedRegionModel()
        model_str = model.__str__()
        assert(type(model_str) == str)

    def test_sample(self):
        """ Tests sampling all random variables.
        """
        model = fcdiff.UnsharedRegionModel()
        N = 10
        H = 5
        U = 4
        C = 45
        (R, T, F, f_tilde, B, b_tilde) = model.sample(N, H, U)
        nptest.assert_equal(R.shape, (N, U))
        nptest.assert_equal(R.dtype, np.dtype('bool'))
        nptest.assert_equal(T.shape, (C, U))
        nptest.assert_equal(T.dtype, np.dtype('bool'))
        nptest.assert_equal(F.shape, (C, 3))
        nptest.assert_equal(F.dtype, np.dtype('bool'))
        nptest.assert_equal(f_tilde.shape, (C, U, 3))
        nptest.assert_equal(f_tilde.dtype, np.dtype('bool'))
        nptest.assert_equal(B.shape, (C, H))
        nptest.assert_equal(B.dtype, np.dtype('float64'))
        nptest.assert_equal(b_tilde.shape, (C, U))
        nptest.assert_equal(b_tilde.dtype, np.dtype('float64'))

    def test_sample_R(self):
        """ Tests sampling R and checks estimated pi.
        """
        model = fcdiff.UnsharedRegionModel()
        r = model.sample_R(3, 10000)
        pi_est = np.mean(r, axis = 1)
        nptest.assert_allclose(pi_est, model.pi, atol = 0.02)

    def test_sample_T_partially_connected(self):
        """ Tests sampling T and given R that implies partial connectivity.
        """
        model = fcdiff.UnsharedRegionModel()
        r = np.tile(np.array([[1], [0], [0]], dtype='bool'), (1, 10000))
        t = model.sample_T(r)
        nptest.assert_equal(t[2, :], 0)
        eta_est = np.mean(t[0:2, :], axis = 1)
        nptest.assert_allclose(eta_est, model.eta, atol = 0.05)

    def test_sample_T_fully_connected(self):
        """ Tests sampling T given R that implies full connectivity.
        """
        model = fcdiff.UnsharedRegionModel()
        r = np.ones((3, 1), dtype = 'bool')
        t = model.sample_T(r)
        t_exp = np.ones((3, 1), dtype = 'bool')
        nptest.assert_array_equal(t, t_exp)

    def test_sample_T_fully_disconnected(self):
        """ Tests sampling T given R that implies full disconnectivity.
        """
        model = fcdiff.UnsharedRegionModel()
        r = np.zeros((3, 1), dtype = 'bool')
        t = model.sample_T(r)
        t_exp = np.zeros((3, 1), dtype = 'bool')
        nptest.assert_array_equal(t, t_exp)

    def test_sample_F_valid(self):
        """ Tests sampling F with valid arguments.
        """
        model = fcdiff.UnsharedRegionModel()
        f = model.sample_F(100)
        gamma_est = np.mean(f, axis = 0)
        nptest.assert_allclose(gamma_est, model.gamma, atol = 0.05)

    def test_sample_F_tilde_typical_negative(self):
        """ Tests sampling F-tilde with a typical negative connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[1, 0, 0]], dtype = 'bool')
        t = np.tile(np.array([0], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([1 - e, e / 2, e / 2])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_F_tilde_typical_neutral(self):
        """ Tests sampling F-tilde with a typical neutral connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 1, 0]], dtype = 'bool')
        t = np.tile(np.array([0], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([e / 2, 1 - e, e / 2])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_F_tilde_typical_positive(self):
        """ Tests sampling F-tilde with a typical positive connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 0, 1]], dtype = 'bool')
        t = np.tile(np.array([0], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([e / 2, e / 2, 1 - e])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_F_tilde_anomalous_negative(self):
        """ Tests sampling F-tilde with an anomalous negative connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[1, 0, 0]], dtype = 'bool')
        t = np.tile(np.array([1], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([e, (1 - e) / 2, (1 - e) / 2])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_F_tilde_anomalous_neutral(self):
        """ Tests sampling F-tilde with an anomalous neutral connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 1, 0]], dtype = 'bool')
        t = np.tile(np.array([1], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([(1 - e) / 2, e, (1 - e) / 2])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_F_tilde_anomalous_positive(self):
        """ Tests sampling F-tilde with an anomalous positive connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 0, 1]], dtype = 'bool')
        t = np.tile(np.array([1], dtype = 'bool'), (1, 10000))
        f_tilde = model.sample_F_tilde(f, t)
        p_est = np.squeeze(np.mean(f_tilde, axis = 1))
        e = model.epsilon
        p_exp = np.array([(1 - e) / 2, (1 - e) / 2, e])
        nptest.assert_allclose(p_est, p_exp, atol = 0.05)

    def test_sample_B_range(self):
        """ Tests that sampling B creates values in [-1, 1].
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            ], dtype = 'bool')
        b = model.sample_B(f, 10000)
        nptest.assert_array_less(b, 1.00001)
        nptest.assert_array_less(-1.00001, b)

    def test_sample_B_negative(self):
        """ Tests sampling B from a negative connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[1, 0, 0]], dtype = 'bool')
        b = model.sample_B(f, 10000)
        mu_est = np.mean(b, axis = 1)
        sigma_est = np.std(b, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[0], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[0], atol = 0.05)

    def test_sample_B_neutral(self):
        """ Tests sampling B from a neutral connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 1, 0]], dtype = 'bool')
        b = model.sample_B(f, 10000)
        mu_est = np.mean(b, axis = 1)
        sigma_est = np.std(b, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[1], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[1], atol = 0.05)

    def test_sample_B_positive(self):
        """ Tests sampling B from a positive connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f = np.array([[0, 0, 1]], dtype = 'bool')
        b = model.sample_B(f, 10000)
        mu_est = np.mean(b, axis = 1)
        sigma_est = np.std(b, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[2], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[2], atol = 0.05)

    def test_sample_B_tilde_range(self):
        """ Tests that sampling b_tilde creates values in [-1, 1].
        """
        model = fcdiff.UnsharedRegionModel()
        f_tilde = np.tile(
                np.array([
                    [[1, 0, 0]],
                    [[0, 1, 0]],
                    [[0, 0, 1]],
                    ], dtype = 'bool'),
                (1, 10000, 1))
        b_tilde = model.sample_B_tilde(f_tilde)
        nptest.assert_array_less(b_tilde, 1 + np.finfo(float).eps)
        nptest.assert_array_less(-1 - np.finfo(float).eps, b_tilde)

    def test_sample_B_tilde_negative(self):
        """ Tests sampling b_tilde from a negative connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f_tilde = np.tile(np.array([[[1, 0, 0]]], dtype = 'bool'), (1, 10000, 1))
        b_tilde = model.sample_B_tilde(f_tilde)
        mu_est = np.mean(b_tilde, axis = 1)
        sigma_est = np.std(b_tilde, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[0], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[0], atol = 0.02)

    def test_sample_B_tilde_neutral(self):
        """ Tests sampling b_tilde from a neutral connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f_tilde = np.tile(np.array([[[0, 1, 0]]], dtype = 'bool'), (1, 10000, 1))
        b_tilde = model.sample_B_tilde(f_tilde)
        mu_est = np.mean(b_tilde, axis = 1)
        sigma_est = np.std(b_tilde, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[1], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[1], atol = 0.02)

    def test_sample_B_tilde_positive(self):
        """ Tests sampling b_tilde from a positive connection.
        """
        model = fcdiff.UnsharedRegionModel()
        f_tilde = np.tile(np.array([[[0, 0, 1]]], dtype = 'bool'), (1, 10000, 1))
        b_tilde = model.sample_B_tilde(f_tilde)
        mu_est = np.mean(b_tilde, axis = 1)
        sigma_est = np.std(b_tilde, axis = 1)
        nptest.assert_allclose(mu_est, model.mu[2], atol = 0.05)
        nptest.assert_allclose(sigma_est, model.sigma[2], atol = 0.02)

