import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks

from sifi_cm.data_fit import FitSigmoid, smooth
from typing import Literal
from collections import namedtuple


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
           direction: Literal["right", "left"] = "right",
           peak_threshold=0.9,
           verbose=False):
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
    peaks = find_peaks(y)[0]
    current_peak = peaks[y[peaks].argmax()]
    if direction == "right":
        candidates = peaks[:y[peaks].argmax()]
    else:
        candidates = peaks[y[peaks].argmax()+1:]
    while any(y[candidates]/y[current_peak] > peak_threshold):
        candidates2 = candidates[y[candidates]/y[current_peak] > peak_threshold]
        current_peak = candidates2[y[candidates2].argmax()]
        if direction == "right":
            candidates = candidates[:y[candidates].argmax()]
        else:
            candidates = candidates[y[candidates].argmax()+1:]
    ymax = y[current_peak]
    if direction == "right":
        ymin = np.min(y[:y.argmax()])
    else:
        ymin = np.min(y[y.argmax():])

    y_50proc = np.mean([ymin, ymax])

    f = interpolate.UnivariateSpline(x, y - y_50proc, s=0)
    roots = f.roots()
    if direction == "right":
        distal_val = roots[roots < x[current_peak]]
    elif direction == "left":
        distal_val = roots[roots > x[current_peak]]
    if distal_val.shape[0] == 0:
        print(distal_val)
        return None
    if len(distal_val) > 1:
        return min(distal_val)
    if verbose:
        result = namedtuple("distal", "distal ymax ymin y50proc xmax xmin")
        return result(distal_val[0], ymax, ymin,
                      y_50proc, x[current_peak],
                      x[y.argmax():][np.argmin(y[y.argmax():])])
    return distal_val[0]
