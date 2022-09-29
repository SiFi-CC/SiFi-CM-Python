from typing import Tuple
import numpy as np
from sifi_cm.root_aux import Edges
from scipy import interpolate
from scipy.signal import find_peaks
from collections import namedtuple
import scipy.optimize as opt
from scipy import ndimage


class Distal50:

    def __init__(self, coordinates: Edges,
                 edge_min_rate=0.6,
                 beam_direction="left"):
        """
        Initialize parameters
        coordinates: Edges
        object of Edges class with coordinates
        edge_min_rate: float
        min rate of edge peak
        beam_direction: str
        """
        self._edges = coordinates
        self._edge_min_rate = edge_min_rate
        self._beam_direction = beam_direction

    def __call__(self,
                 projection: np.ndarray) -> Tuple[float, float, float, float]:
        """calculate distall fall-off position with '50%' method

        Parameters
        ----------
        projection : npt.ArrayLike
            1D projection of the reconstructed image

        Returns
        -------
        Tuple[float, float, float, float]
            50% peak position, value and edge peak position, value
        """
        if len(projection.shape) > 1:
            raise ValueError("should be 1D")
        if projection.shape[0] != self._edges.x_cent.shape[0]:
            raise ValueError("should be the same size as coordinates")
        edge_index = 0
        edge_peak = 0
        if self._beam_direction == "right":
            projection = projection[::-1]
            self._edges = Edges(self._edges.x[::-1], self._edges.y)
        # get edge peak
        peaks = find_peaks(projection)[0]
        total_max = projection[peaks].max()
        # print(total_max, self._edge_min_rate)
        # take the most left peak larger then total_max * self._edge_min_rate
        edge_index = peaks[projection[peaks]
                           > total_max * self._edge_min_rate][0]
        edge_peak = projection[edge_index]
        # print(edge_peak, edge_index)
        # interpolation
        f = interpolate.InterpolatedUnivariateSpline(
            self._edges.x_cent[:edge_index],
            projection[:edge_index] - 0.5*edge_peak)
        roots = f.roots()
        return (roots[-1], 0.5*edge_peak,
                self._edges.x_cent[edge_index], edge_peak)


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


def Gaussian_1D(x, mean, sigma,  amplitude, cut):
    "PDF for the Normal distribution"
    return cut + amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def Gaussian_2D(xdata_tuple,
                meanx, meany,
                sigmax, sigmay, theta,
                amplitude, cut):
    """Probability density function for the 2D multivariate normal distribution
    with custom amplitude and shift along Y.
    This function is used for fitting.

    Parameters
    ----------
    xdata_tuple : tuple(numpy.array, numpy.array)
        Tuple of the X and Y coordinates
    meanx : float
            Location parameter for X axis
    meany : float
            Location parameter for Y axis
    sigmax : float
             Standard deviation for X axis
    sigmay : float
             Standard deviation for Y axis
    theta : float
            angle of the ellipsoid
    amplitude : float
    cut : float

    Returns
    -------
    numpy.array
        Flattened array of the function values
    """
    x, y = np.meshgrid(*xdata_tuple)
    a = (np.cos(theta)**2)/(2*sigmax**2) + (np.sin(theta)**2)/(2*sigmay**2)
    b = -(np.sin(2*theta))/(4*sigmax**2) + (np.sin(2*theta))/(4*sigmay**2)
    c = (np.sin(theta)**2)/(2*sigmax**2) + (np.cos(theta)**2)/(2*sigmay**2)
    g = cut + amplitude*np.exp(- (a*((x-meanx)**2) + 2*b*(x-meanx)*(y-meany)
                                  + c*((y-meany)**2)))
    return g.ravel()


def fit_1d(x, data):
    # first guess
    mean = sum(x * data) / sum(data)
    sigma = np.sqrt(sum(data * (x - mean) ** 2) / sum(data))
    popt, _ = opt.curve_fit(Gaussian_1D, x, data,
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
    popt, _ = opt.curve_fit(
        Gaussian_2D, (x, y), data.ravel(), (meanx, meany, 1, 1, 1, 1, 0))
    return GAUSS2D(*popt)


def normalize(x):
    if np.unique(x).shape[0] > 1:
        return (x - x.min())/(x.max() - x.min())
    else:
        return np.ones_like(x)


# def smooth(data, scale=7, norm=True, filter="median"):
#     if filter == "median":
#         smoothed = ndimage.median_filter(data, size=scale)
#     elif filter == "gaus" or "gauss":
#         smoothed = ndimage.gaussian_filter(data, scale)
#     else:
#         raise ValueError("this filter is not defined")
#     if norm:
#         smoothed -= np.min(smoothed)
#         smoothed /= smoothed.max()
#     return smoothed
