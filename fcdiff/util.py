"""
Provides convenient, but non-vital utilities.
"""

import numpy as np

def N_to_C(N):
    """
    Computes the number of connections for a network with N regions.

    Arguments
    ---------
    N : 2 <= int
        Number of regions.

    Returns
    -------
    C : 1 <= int
        Number of connections.
    """
    return N * (N - 1) / 2

def C_to_N(C):
    """
    Computes the number of regions for a network with C connections.

    Arguments
    ---------
    C : 1 <= int
        Number of connections.

    Returns
    -------
    N : 1 <= int
        Number of regions.
    """
    N = (np.sqrt(8 * C + 1) - 1) / 2 + 1
    return N

def nm_to_c(n, m):
    """
    Converts a pair of region indices (n, m) to a connection index.

    Arguments
    ---------
    n : 0 <= int
        First region index.
    m : n < int
        Second region index.

    Returns
    -------
    c : 0 <= int < N * (N - 1) / 2
        Connection index.

    Notes
    -----
    N is the total number of regions.
    """
    return N_to_C(n) + m

def c_to_nm(c):
    """
    Converts a connection index to a pair of region indices (n, m).

    Arguments
    ---------
    c : 0 <= int < N * (N - 1) / 2
        Connection index.

    Returns
    -------
    n : 0 <= int
        First region index.
    m : n < int
        Second region index.

    Notes
    -----
    N is the total number of regions.
    """
    n = np.floor((np.sqrt(8 * c + 1) - 1) / 2) + 1
    m = c - N_to_C(n)
    return (n, m)
