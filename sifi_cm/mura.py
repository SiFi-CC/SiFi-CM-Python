import numpy as np


def is_prime(n):
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n < 2:
        return False
    for i in range(3, int(n**0.5) + 1, 2):   # only odd numbers
        if n % i == 0:
            return False
    return True


def get_mura(order):
    """Return 2D MURA array
    It is not a classical way.
    Comparing to the wikipedia algorytm the mask is inverted
    """
    if not is_prime(order):
        raise ValueError("The mask order should be prime")
    quadratic_residues = np.unique(np.arange(order)**2 % order)
    mask = np.zeros((order, order))
    c = -np.ones(order)
    c[np.isin(np.arange(order), quadratic_residues)] = 1
    cc = np.outer(c, c)
    mask[np.where(cc == 1)] = 1
    mask = np.abs(mask - 1)[:, ::-1]
    mask[0, :] = 1
    mask[:, -1] = 0
    return mask


def get_mura_cut(order, cutx=None, cuty=None, one_dim=False):
    """Return 2D MURA array
    It is not a classical way.
    Comparing to the wikipedia algorytm the mask is inverted
    """
    if cutx and not cuty:
        cuty = cutx
    if one_dim:
        mask = get_mura_1d(order)
    else:
        mask = get_mura(order)
    print(mask.shape)
    if cutx and cuty:
        mask = mask[order//2-cuty//2:order//2+cuty // 2 + 1,
                        order//2-cutx//2:order//2+cutx//2 + 1]
    if not one_dim:
        mask[0, :] = 1
        mask[:, -1] = 0
    return mask


def get_mura_1d(order):
    """Return 1D MURA array
    It is not a classical way.
    Comparing to the wikipedia algorytm the mask is inverted
    """
    if not is_prime(order):
        raise ValueError("The mask order should be prime")
    quadratic_residues = np.unique(np.arange(order)**2 % order)
    mask = np.zeros(order)
    c = -np.ones(order)
    c[np.isin(np.arange(order), quadratic_residues)] = 1
    # cc = np.outer(c, c)
    mask[np.where(c == 1)] = 1
    mask = np.abs(mask - 1)[::-1]
    mask = np.repeat(mask[:, np.newaxis], order, axis=1).T
    return mask
