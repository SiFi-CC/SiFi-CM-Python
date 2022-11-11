from collections import namedtuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from sifi_cm.functions import Gaussian_1D, Gaussian_2D, sigmoid3


class GAUSS1D(namedtuple("gauss_params", "mean sigma amplitude cut")):
    @property
    def variance(self):
        return self.sigma**2


class GAUSS2D(namedtuple("gauss2D_params",
                         "meanx meany sigmax sigmay theta ampl cut")):
    @property
    def variancex(self):
        return self.sigmax**2

    @property
    def variancey(self):
        return self.sigmay**2


def fit_1d(x, data):
    "Fit 1D array with Gaussian"
    # first guess
    mean = sum(x * data) / sum(data)
    sigma = np.sqrt(sum(data * (x - mean) ** 2) / sum(data))
    popt, _ = curve_fit(Gaussian_1D, x, data,
                        (mean, sigma, max(data), min(data)))
    return GAUSS1D(*popt)


def fit_2d(x, y, data):
    """Fit the function values with 2D Gaussian

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
    data : numpy.array
        2D numpy array with function values

    Returns
    -------
    GAUSS2D
        fitting parameters
    """
    meanx = x[np.argmax(data.sum(axis=0))]
    meany = y[np.argmax(data.sum(axis=1))]
    popt, _ = curve_fit(
        Gaussian_2D, (x, y), data.ravel(), (meanx, meany, 1, 1, 1, 1, 0))
    return GAUSS2D(*popt)


def smooth(data, scale=7, norm=True, filter="median"):
    if filter == "median":
        smoothed = ndimage.median_filter(data, size=scale)
    elif filter == "gaus" or "gauss":
        smoothed = ndimage.gaussian_filter(data, scale)
    else:
        raise ValueError("this filter is not defined")
    if norm:
        smoothed = normalize(smoothed)
    return smoothed


def normalize(x):
    if np.unique(x).shape[0] > 1:
        return (x - x.min())/(x.max() - x.min())
    else:
        return np.ones_like(x)


def FitSigmoid(fit_range, profile):
    """Fit Sigmoid to function."""
    # peaks = find_peaks(profile)[0]
    maxpeak_index = profile.argmax()
    left_x = fit_range[:maxpeak_index]
    left_y = profile[:maxpeak_index]
    maxgrad = left_x[np.gradient(left_y).argmax()]
    # print("first guess inflection:", round(maxgrad, 2))
    p, _ = curve_fit(sigmoid3, fit_range, profile,
                     # p0=[profile.max(),1,fit_range[0],0.1],
                     p0=[profile.max(), 1,
                         maxgrad, 0],
                     maxfev=10000,
                     method='trf')
    return p
