import sys, os
import numpy.testing as nptest
import fcdiff.util

def test_N_to_C_to_N():
    """ Tests converting from N to C back to N.
    """
    for N in range(2, 10):
        C = fcdiff.util.N_to_C(N)
        N2 = fcdiff.util.C_to_N(C)
        msg = "Converted from %u to %u back to %u." % (N, C, N2)
        nptest.assert_equal(N2, N, err_msg = msg)

def test_nm_to_c():
    """ Tests converting pair of region indices to a connection index.
    """
    N = 10
    c = 0
    for n in range(N):
        for m in range(n):
            act_c = fcdiff.util.nm_to_c(n, m)
            msg = "Converted (%u, %u) to %u not %u." % (n, m, act_c, c)
            nptest.assert_equal(act_c, c, err_msg = msg)
            c += 1

def test_c_to_nm():
    """ Tests converting pair of region indices to a connection index.
    """
    N = 10
    c = 0
    for n in range(N):
        for m in range(n):
            (act_n, act_m) = fcdiff.util.c_to_nm(c)
            msg = "Converted %u to (%u, %u) not (%u, %u)." % (c, act_n, act_m, n, m)
            nptest.assert_equal((act_n, act_m), (n, m), err_msg = msg)
            c += 1

