"""
Probabilistic models that generate correlations.
"""

import numpy as np
import textwrap
import fcdiff

class UnsharedRegionModel(object):
    """
    The individual anomalous region (IAR) model.

    Attributes
    ----------
    rng : :class:`numpy.random.RandomState`
        Random number generator.
    pi : 0 <= float <= 1
        Probability of an anomalous region.
    eta : 0 <= float <= 1
        Probability of an anomalous connection btw a typical and anomalous region.
    gamma : :class:`numpy.ndarray`, (3,), 0 <= float <= 1
        Probability of each template connection type (array sums to 1).
    epsilon : 0 <= float <= 1
        Probability that a typical connection differs from the template.
    mu : :class:`numpy.ndarray`, (3,), -1 <= float <= 1
        Mean correlation of each connection type.
    sigma : :class:`numpy.ndarray`, (3,), 0 < float
        Standard deviation of the correlation of each connection type.
    """

    def __init__(self):
        self.rng = np.random.RandomState(0)
        self.pi = 0.05
        self.eta = 0.3
        self.gamma = np.array([0.1, 0.8, 0.1])
        self.epsilon = 0.03
        self.mu = np.array([-0.15, 0, 0.3])
        self.sigma = np.array([0.025, 0.035, 0.05])

    def __str__(self):
        return textwrap.dedent('''\
            fcdiff.models.UnsharedRegionModel
                rng = %s
                pi = %g
                eta = %g
                gamma = %s
                epsilon = %g
                mu = %s
                sigma = %s''' % (self.rng, self.pi, self.eta, self.gamma,
                    self.epsilon, self.mu, self.sigma))

    def sample(self, N, H, U):
        """
        Samples all random variables from the model.

        Arguments
        ---------
        N : 2 <= int
            Number of regions.
        H : 1 <= int
            Number of healthy subjects.
        U : 1 <= int
            Number of unhealthy patients.

        Returns
        -------
        r : :class:`numpy.ndarray`, (N, U), bool
            Anomalous regions of unhealthy patients.
        t : :class:`numpy.ndarray`, (C, U), bool
            Anomalous connections of unhealthy patients.
        f : :class:`numpy.ndarray`, (C, 3), bool
            Connection template of healthy subjects.
        f_tilde : :class:`numpy.ndarray`, (C, U, 3), bool
            Connections of unhealthy patients.
        b : :class:`numpy.ndarray`, (C, H), -1 <= float <= 1
            Correlations of healthy subjects.
        b_tilde : :class:`numpy.ndarray`, (C, U), -1 <= float <= 1
            Correlations of unhealthy patients.

        Notes
        -----
        C is the number of connections which is :math:`N (N - 1) / 2`.
        """
        r = self.sample_R(N, U)
        t = self.sample_T(r)
        f = self.sample_F(N)
        f_tilde = self.sample_F_tilde(f, t)
        b = self.sample_B(f, H)
        b_tilde = self.sample_B_tilde(f_tilde)
        return (r, t, f, f_tilde, b, b_tilde)

    def sample_R(self, N, U):
        """
        Samples anomalous regions in unhealthy patients.

        Arguments
        ---------
        N : 2 <= int
            Number of regions.
        U : 1 <= int
            Number of unhealthy patients.

        Returns
        -------
        r : :class:`numpy.ndarray` (N, U), bool
            Anomalous regions in unhealthy patients.
        """
        r = self.rng.binomial(1, self.pi, (N, U)) > 0
        return r

    def sample_T(self, r):
        """
        Samples anomalous connections given anomalous regions.

        Arguments
        ---------
        r : :class:`numpy.ndarray`, (N, U), bool
            Anomalous regions of unhealthy patients.

        Returns
        -------
        t : :class:`numpy.ndarray`, (C, U), bool
            Anomalous connections of unhealthy patients.

        Notes
        -----
        C is the number of connections which is :math:`N (N - 1) / 2`.
        """
        (N, U) = r.shape
        C = fcdiff.N_to_C(N)
        t = np.zeros((C, U), dtype = 'bool')
        c = 0
        for n in range(N):
            for m in range(n + 1, N):
                for u in range(U):
                    if not r[n, u] and not r[m, u]:
                        t[c, u] = False
                    elif r[n, u] and r[m, u]:
                        t[c, u] = True
                    else:
                        t[c, u] = self.rng.binomial(1, self.eta) > 0
                c += 1
        return t

    def sample_F(self, N):
        """
        Samples a connection template of healthy subjects.

        Arguments
        ---------
        N : 2 <= int
            Number of regions.

        Returns
        -------
        f : :class:`numpy.ndarray`, (C, 3), bool
            Connection template of healthy subjects.
        """
        C = fcdiff.N_to_C(N)
        return self.rng.multinomial(1, self.gamma, C) > 0

    def sample_F_tilde(self, f, t):
        """
        Samples patients' connections given a template and anomalous connections.

        Arguments
        ---------
        f : :class:`numpy.ndarray`, (C, 3), bool
            Connection template of healthy subjects.
        t : :class:`numpy.ndarray`, (C, U), bool
            Anomalous connections of unhealthy patients.

        Returns
        -------
        f_tilde : :class:`numpy.ndarray`, (C, U, 3), bool
            Connections of unhealthy patients.
        """
        (C, U) = t.shape
        f_tilde = np.zeros((C, U, 3), dtype = 'bool')
        e = self.epsilon
        for c in range(C):
            pT0 = (1 - e) * f[c, :] + (e * 0.5) * (1 - f[c, :])
            pT1 = e * f[c, :] + (1 - e) * 0.5 * (1 - f[c, :])
            for u in range(U):
                if t[c, u]:
                    f_tilde[c, u, :] = self.rng.multinomial(1, pT1) > 0
                else:
                    f_tilde[c, u, :] = self.rng.multinomial(1, pT0) > 0
        return f_tilde

    def sample_B(self, f, H):
        """
        Samples correlations of healthy subjects given a connection template.

        Arguments
        ---------
        f : :class:`numpy.ndarray`, (C, 3), bool
            Connection template of healthy subjects.
        H : 1 <= int
            Number of healthy subjects.

        Returns
        -------
        b : :class:`numpy.ndarray`, (C, H), -1 <= float64 <= 1
            Correlations of healthy subjects.
        """
        C = f.shape[0]
        b = np.zeros((C, H), dtype = 'float64')
        for c in range(C):
            m = self.mu[f[c, :]]
            s = self.sigma[f[c, :]]
            b[c, :] = self.rng.normal(m, s, H)
        return b.clip(-1, 1)

    def sample_B_tilde(self, f_tilde):
        """
        Samples correlations of unhealthy patients given their connections.

        Arguments
        ---------
        f_tilde : :class:`numpy.ndarray`, (C, U, 3), bool
            Connections of unhealthy patients.

        Returns
        -------
        b_tilde : :class:`numpy.ndarray`, (C, U), -1 <= float64 <= 1
            The correlations of unhealthy patients.
        """
        (C, U) = f_tilde.shape[0:2]
        b_tilde = np.zeros((C, U), dtype = 'float64')
        for c in range(C):
            for u in range(U):
                m = self.mu[f_tilde[c, u, :]]
                s = self.sigma[f_tilde[c, u, :]]
                b_tilde[c, u] = self.rng.normal(m, s)
        return b_tilde.clip(-1, 1)

