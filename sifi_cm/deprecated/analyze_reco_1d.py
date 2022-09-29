import matplotlib.pyplot as plt
import root_aux as raux
import numpy as np
# import scipy
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit


def Gauss(x, H, A, x0, sigma):
    "PDF for the Normal distribution"
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def get_reco(filenames, coordinates, iter=100):
    """
    get reconstruction histograms from all files from filenames from

    filenames: iterable
    coordinates: iterable
    iter: int
    number of iterations in reconstructions

    Return:
    dictionary with key - source coordinate, value - histogram
    """
    reco = {}
    if len(coordinates) != len(filenames):
        raise KeyError("Different sizes")
    for s, filename in zip(coordinates, filenames):
        reco[s] = raux.get_histo(filename, [f"reco;{iter}"])[0]
    return reco


def fit_all(histograms):
    """Fit all histograms with gaussian

    Parameters
    ----------
    histograms : dictionary
        key - source coordinate, value - histogram

    Returns
    -------
    tuple
        pair of two numpy arrays: mean values and variances.
        Each array has two columns: fitted value and standard deviation. 
    """
    mean_values = []
    sigma_values = []
    for s in histograms:
        # print(s)
        binWidth = (histograms[s].edges.x[1]-histograms[s].edges.x[0])/2
        x = histograms[s].edges.x[:-1] + binWidth
        y = histograms[s].vals.flatten()
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(Gauss, x, y, p0=[min(y), max(y), mean, sigma])
        perr = np.sqrt(np.diag(pcov))
        mean_values.append([popt[2], perr[2]])
        sigma_values.append([popt[3], perr[3]])
    return np.array(mean_values), np.array(sigma_values)


if __name__ == '__main__':
    coordinates = np.arange(-30, 30, 0.5)
    filenames = [f"./1d_map/reco220_170_n1e6_det32_2lay_nowallpet_mask31_70mm_1d_{str(s)}_0.root" for s in coordinates]
    # filenames = [f"./1d_map_467/reco220_170_n1e6_det32_2lay_nowallpetcut31_mask467_70mm_1d_{str(s)}_0.root" for s in coordinates]
    reco = get_reco(filenames, coordinates, iter=200)

    binWidth = (reco[coordinates[0]].edges.x[1]-reco[coordinates[0]].edges.x[0])/2
    colors = list(mcolors.BASE_COLORS.keys())

    # Make plot of fitted pictures for selected source positions
    x_space = np.linspace(reco[coordinates[0]].edges.x[0], reco[coordinates[0]].edges.x[-1],10000)
    plt.figure(6, figsize=(20, 10))
    i = 0
    for i, s in enumerate(np.arange(-16.5, -14.5, 0.5)):
        x = reco[s].edges.x[:-1] + binWidth
        y = reco[s].vals.flatten()
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(Gauss, x, y, p0=[min(y), max(y), mean, sigma])
        print("for x=", s, "parameters: ", popt)
        y_space = Gauss(x_space, *popt)
        print("for x=", s, "argmax gives :", x_space[np.argmax(y_space)])
        plt.plot(x_space, y_space, label=s, c=colors[i])
        # plt.plot(x, y, "o-", label=s,  c=colors[i])
        plt.vlines(x=popt[2] , ymin=0, ymax=np.max(y_space), ls="--", color=colors[i])
        plt.text(popt[2], 200*i ,f'{round(popt[2],2)}',rotation=45, size=18, color=colors[i])

    plt.minorticks_on()
    plt.grid(which="both")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.legend(title="source x coordinate", fontsize=16, title_fontsize=16)
    plt.title("Reconstruction 200 iterations smoothed with gaussian filter (kernel = 2[bins])", fontsize=18)
    # plt.savefig("1d_reco_fitted.png", facecolor='white')
    # plt.show()

    mean_values, sigma_values = fit_all(reco)
    plt.figure(1, figsize=(20,10))
    plt.plot(coordinates, coordinates, "--", c="r")
    plt.plot(coordinates, mean_values[:,0], lw=2)
    plt.title("200 iterations")
    plt.xlabel("true source position", fontsize=18)
    plt.ylabel("reconstructed source position", fontsize=18)
    # plt.savefig("1d_map-fitted-mean_true.png", facecolor="white")
    # plt.show()

    plt.figure(2, figsize=(20,10))
    plt.plot(coordinates, np.abs(sigma_values[:,0]), lw=2)
    plt.title("200 iterations", fontsize=18)
    plt.xlabel("true source position [mm]", fontsize=18)
    plt.ylabel("$\sigma_{fit}$", fontsize=18)
    # plt.savefig("1d_map-fitted-sigma_true.png", facecolor="white")
    # plt.show()

    plt.figure(3, figsize=(20,10))
    plt.plot(coordinates, -coordinates+mean_values[:,0], lw=2)
    plt.title("200 iterations", fontsize=18)
    plt.ylabel("reconstructed - true [mm]", fontsize=18)
    plt.xlabel("true source position [mm]", fontsize=18)
    # plt.savefig("1d_map-fitted-error_true.png", facecolor="white")
    # plt.show()

    # plt.figure(4, figsize=(20,10))
    # plt.plot(coordinates, np.abs(coordinates-mean_values[:,0]), lw=2)
    # plt.title("200 iterations", fontsize=18)
    # plt.ylabel("$\mid reconstructed - true \mid$ [mm]", fontsize=18)
    # plt.xlabel("true source position [mm]", fontsize=18)
    # plt.savefig("1d_map-fitted-error_true_abs.png", facecolor="white")
    # plt.show()

    plt.figure(5, figsize=(20,10))
    plt.hist(coordinates-mean_values[:,0], bins=50)
    plt.xlabel("true - reconstructed [mm]", fontsize=18)
    # plt.savefig("1d_map-fitted-error_dist.png", facecolor="white")
    plt.show()
