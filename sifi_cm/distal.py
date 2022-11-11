import numpy as np
from data_fit import smooth
from scipy import interpolate
from scipy.optimize import curve_fit


def sigmoid3(x, L, b, x0, k):
    """Sigmoid function."""
    y = L / (1 + np.exp(-k*(x-x0), dtype=np.float128)) + b
    return (y)


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


def sigmoid(x, y, direction="right", range_deltax=15):
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


def spline(x, y, filter_name="gaussian",
           filter_scale=3, direction="right"):
    ymax = y.max()
    ymin = np.median(y[:y.argmax()])  # ???
    y_50proc = np.mean([ymin, ymax])

    ysm = smooth(y, filter=filter_name, scale=filter_scale)
    f = interpolate.UnivariateSpline(x, ysm - y_50proc, s=0)
    roots = f.roots()
    distal = roots[roots < x[y.argmax()]]
    if len(distal) > 1:
        print("Warning! More than 1 DFPD")
    return distal[0]
