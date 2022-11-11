import numpy as np 


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


def sigmoid3(x, L, b, x0, k):
    """Sigmoid function."""
    y = L / (1 + np.exp(-k*(x-x0), dtype=np.float128)) + b
    return (y)