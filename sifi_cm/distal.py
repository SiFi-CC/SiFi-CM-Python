import numpy as np
from scipy import interpolate

from sifi_cm.data_fit import FitSigmoid, smooth
from typing import Literal


def sigmoid(x: np.ndarray, y: np.ndarray,
            direction: Literal["right", "left"] = "right",
            range_deltax: float = 15):
    """Find a DFPD basing on the sigmoid fitting

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    direction : str, optional
        direction of the beam, by default "right"
    range_deltax : float, optional
        range of fitting to the right of the peak
        and to the left of minimum value
        (other way round if direction=='left'), by default 15

    Returns
    -------
    float
        DFPD
    """
    xmax = x[y.argmax()]
    if direction == "right":
        xmin = x[y[:y.argmax()].argmin()]
        indeces = np.where((x <= xmax + range_deltax) &
                           (x >= xmin - range_deltax))
        p_fit = FitSigmoid(x[indeces], y[indeces])
    else:
        indeces = np.where((x > xmax - range_deltax))
        p_fit = FitSigmoid(x[indeces][::-1], y[indeces][::-1])
    return p_fit[2]


def spline(x: np.ndarray, y: np.ndarray,
           filter_name: Literal["gaussian", "median", None] = "gaussian",
           filter_scale: int = 3,
           direction: Literal["right", "left"] = "right"):
    """Find a DFPD basing on the
    procedure described in DOI:10.1088/0031-9155/58/13/4563

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    filter_name : str, optional
        Smoothing applied to the data, either 'gaussian' or 'median',
        by default "gaussian"
    filter_scale : int, optional
        scale of filter, by default 3
    direction : str, optional
        direction of the beam, by default "right"

    Returns
    -------
    float
        DFPD
    """
    if filter_name:
        y = smooth(y, filter=filter_name, scale=filter_scale)
    ymax = y.max()
    if direction == "right":
        ymin = np.median(y[:y.argmax()])  # ???
    else:
        ymin = np.median(y[y.argmax():])
    y_50proc = np.mean([ymin, ymax])

    f = interpolate.UnivariateSpline(x, y - y_50proc, s=0)
    # print("HI")
    roots = f.roots()
    if direction == "right":
        distal_val = roots[roots < x[y.argmax()]]
    elif direction == "left":
        distal_val = roots[roots > x[y.argmax()]]
    if distal_val.shape[0] == 0:
        print(distal_val)
        return None
        # raise Exception("Error")
    if len(distal_val) > 1:
        return min(distal_val)
    return distal_val[0]
