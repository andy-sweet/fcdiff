import numpy as np

class OneSampleModel:
    """
    A model of a one sample deviation from a null distribution.
    """

    rng = np.random.RandomState()
    pi = 0.05
    gamma = 0.2
    mu = 0.3
    sigma = 0.05
    epsilon = 0.02
    eta = 0.3

    def sample(self, N, H, U):
        """
        Samples from the model.

        Arguments
        ---------
        N : int
            The number of regions.
        H : int
            The number of healthy subjects.
        U : int
            The number of unhealthy subjects.

        Returns
        -------
        R : NxUx2 ndarray
            The abnormal region indicators of unhealthy subjects.
        T : CxUx2
            The abnormal network indicators of unhealthy subjects.
        F : CxK ndarray
            The connectivity indicators of healthy group.
        Ft : CxUxK
            The connectivity indicators of the unhealthy subjects.
        B : CxH ndarray
            The correlation coefficients of unhealthy subjects.
        Bt : CxU ndarray
            The correlation coefficients of healthy subjects.
        """
        R = self.sample_R(N, U)
        T = self.sample_T(N, U)
        F = self.sample_F(N)
        Ft = self.sample_Ft(N, U)
        B = self.sample_B(N, H)
        Bt = self.sample_Bt(N, U)
        return (R, T, F, Ft, B, Bt)

    def sample_R(self, N, U):
        """
        Samples region indicators.

        Arguments
        ---------
        N : int
            The number of regions.
        U : int
            The number of unhealthy subjects.

        Returns
        -------
        R : NxUx2 ndarray
            The abnormal region indicators of unhealthy subjects.
        """
        return self.rng.multinomial(1, [1-self.pi, self.pi], (N, U))

