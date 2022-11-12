"""
Classes and functions which help to import data from .root files
and to reconstruct image using MLEM algorithm
"""
from collections import deque, namedtuple

import numpy as np
import uproot
from tqdm import tqdm

from sifi_cm.data_fit import normalize
from typing import Union, List


class Edges(namedtuple("XY", "x y")):
    """Class inherited from named tuple which represents
    edges of the histogram coordinates.

    Raises
    ------
    ValueError
        If trying to get binWidth when widths are different for X and Y
    """
    def __new__(cls, x, y, edges=True):
        x = np.array(x)
        y = np.array(y)
        x.sort()
        y.sort()
        if edges:
            return super().__new__(cls, x, y)
        else:
            xstep = x[1]-x[0]
            x_edges = x - 0.5*xstep
            x_edges = np.append(x_edges, x_edges[-1]+xstep)
            ystep = y[1]-y[0]
            y_edges = y - 0.5*ystep
            y_edges = np.append(y_edges, y_edges[-1]+ystep)
            return super().__new__(cls, x_edges, y_edges)

    @property
    def x_binWidth(self):
        return 0.5*abs(self.x[1] - self.x[0])

    @property
    def y_binWidth(self):
        return 0.5*abs(self.y[1] - self.y[0])

    @property
    def binWidth(self):
        if self.x_binWidth == self.y_binWidth:
            return self.x_binWidth
        else:
            raise ValueError("Different widths for X and Y")

    @property
    def x_cent(self):
        return self.x[:-1] - sign(self.x[1] - self.x[0])*self.x_binWidth

    @property
    def y_cent(self):
        return self.y[:-1] - sign(self.y[1] - self.y[0])*self.y_binWidth

    def __eq__(self, other):
        return np.all(self.x == other.x) and np.all(self.y == other.y)

    def xy_edges(self):
        return self.x, self.y

    def xy(self):
        return self.x_cent, self.y_cent


def sign(x):
    return (1, -1)[x > 0]


class Histogram(namedtuple('Histogram', ['vals', 'edges', 'name'])):

    vals: np.ndarray
    edges: Edges
    name: str

    def __repr__(self):
        return "Histogram {}".format(self.name)

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return Histogram(other + self.vals, self.edges, self.name)
        if self.edges == other.edges:
            total_vals = self.vals + other.vals
        else:
            raise KeyError
        if self.name == other.name:
            name = self.name
        else:
            name = "summed_histo"
        return Histogram(total_vals, self.edges, name)

    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            return Histogram(other + self.vals, self.edges, self.name)
        if self.edges == other.edges:
            total_vals = self.vals + other.vals
        else:
            raise KeyError
        if self.name == other.name:
            name = self.name
        else:
            name = "summed_histo"
        return Histogram(total_vals, self.edges, name)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex))\
                and not isinstance(other, bool):
            return Histogram(other*self.vals, self.edges, self.name)
        else:
            raise KeyError

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex))\
                and not isinstance(other, bool):
            return Histogram(other*self.vals, self.edges, self.name)
        else:
            raise KeyError


def get_histo(path, histo_names=["energyDeposits", "sourceHist"],
              edges=True) -> Union[Histogram, List[Histogram]]:
    """Get histogram(s) from the .root file

    Args:
        path (str): Path to the root file
        histo_names (str or iterable of strings): Name(s) of histograms

    Returns:
        if edges:
            list[Histogram] or Histogram: 'Histogram' objects
        else:
            list[np.ndarray] or np.ndarray: histogram values only
    """
    histo_names_it = [histo_names] if type(histo_names) == str else histo_names
    result = []
    with uproot.open(path) as file:
        for name in histo_names_it:
            vals, edgesx, edgesy = file[name].to_numpy()
            if edges:
                result.append(Histogram(vals[:, ::-1],
                                        Edges(edgesx, edgesy[::-1]),
                                        name))
            else:
                if vals.shape[-1] == 1:
                    result.append(vals.flatten())
                else:
                    result.append(vals[:, ::-1])

    return result[0] if type(histo_names) == str else result


def get_deposits(path: str) -> Histogram:
    return get_histo(path, "energyDeposits")


def get_source_edges(path: str) -> Edges:
    return get_histo(path, "sourceHist").edges


def get_hypmed_sim_row(path: str):
    simdata = get_histo(
        path, [f"energyDepositsLayer{i}" for i in range(3)])
    return np.hstack([sim.vals.flatten() for sim in simdata])


def get_deposits_source(path: str) -> List[Histogram]:
    return get_histo(path, ["energyDeposits", "sourceHist"])


def get_hmat(path: str, norm=True, hypmed=False) -> np.ndarray:
    """Get system matrix from .root file. System matrix is supposed to
    be a matrix called 'matrixH'  if 'hypmed == False' and
    3 matrices 'matrixH{0,1,2}' if 'hypmed == True'

    Parameters
    ----------
    path : str
        Path to .root file with system matrix
    norm : bool, optional
        If True, each element of the matrix will be divided
        by the sum of all the values in the corresponding row, by default True
    hypmed : bool, optional
        If True, 3 matrices will be loaded (for each layer of hypmed crystals).
        By default False

    Returns
    -------
    np.array
        System matrix where each row number corresponds to the detector needle,
        while the column specifies the point in the field of view.
    """
    if hypmed:
        with uproot.open(path) as matr_file:
            Tmatr2 = matr_file["matrixH0"]
            matrixH0 = np.array(Tmatr2.member("fElements"))\
                .reshape(Tmatr2.member("fNrows"), Tmatr2.member("fNcols"))

            Tmatr2 = matr_file["matrixH1"]
            matrixH1 = np.array(Tmatr2.member("fElements"))\
                .reshape(Tmatr2.member("fNrows"), Tmatr2.member("fNcols"))

            Tmatr2 = matr_file["matrixH2"]
            matrixH2 = np.array(Tmatr2.member("fElements"))\
                .reshape(Tmatr2.member("fNrows"), Tmatr2.member("fNcols"))
        matrixH = np.vstack((matrixH0, matrixH1, matrixH2))
    else:
        with uproot.open(path) as matr_file:
            Tmatr = matr_file["matrixH"]
            matrixH = np.array(Tmatr.member("fElements"))\
                .reshape(Tmatr.member("fNrows"), Tmatr.member("fNcols"))
    if norm:
        matrixH = matrixH/matrixH.sum(axis=0)  # normalization
    return matrixH


def reco_mlem(matr: np.ndarray, image: np.ndarray,
              niter: int = 100,
              S: np.ndarray = None,
              bg: np.ndarray = None,
              keep_all: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    """LM-MLEM reconstruction

    Parameters
    ----------
    matr : np.ndarray
        System matrix
    image : np.ndarray
        Image vector to be reconstructed
    S : np.ndarray, optional
        Sensetivity map, by default uniform map is used
    bg : np.ndarray, optional
        Background. If used, it is added to the denomenator. By default 0
    niter : int, optional
        Number of iterations, by default 100
    keep_all : bool, optional
        if True - all intermediate reconstructed images
        will be returned.If False - only the last one. By default False
    Returns
    -------
    np.array
        Reconstructed object vector.

    Raises
    ------
    ValueError
        If shapes of vectors/matrix are not appropriate.
    """
    if not isinstance(S, np.ndarray):
        S = matr.sum(axis=0)
    if not isinstance(bg, np.ndarray):
        bg = np.zeros_like(image)
    if matr.shape[0] != image.shape[0] != bg.shape[0]\
            or matr.shape[-1] != S.shape[0]:
        raise ValueError("The shape of vectors are not correct")
    if keep_all:
        maxlen = None
    else:
        maxlen = 2
    reco = deque([np.ones(matr.shape[-1])], maxlen=maxlen)
    for _ in tqdm(range(len(reco)-1, niter), desc="Reconstruction"):
        reco_tmp = reco[-1]/S*(matr.T @ (image/(matr @ reco[-1]+bg)))
        reco.append(reco_tmp)
    return list(reco) if keep_all else reco[-1]


def mse_uqi(x, y, normx=False, normy=False, normall=False):
    """Compute MSI and UQI similarities between 2 vectors

    Parameters
    ----------
    x : np.array
    y : np.array
    normx : bool, optional
        If True - vector x will be normalized, by default False
    normy : bool, optional
        If True - vector y will be normalized, by default False
    normall : bool, optional
        Setting both normx and normy, by default False

    Returns
    -------
    tuple(float, float, float)
        MSE, 1/UQI

    Raises
    ------
    ValueError
        If lenghts of vectors are not the same
    """
    if normx or normall:
        x = normalize(x)
    if normy or normall:
        y = normalize(y)
    if x.shape != y.shape:
        raise ValueError("Shapes of arrays are different")
    mse = ((y - x)**2).mean()
    cov = np.cov(x.flatten(), y.flatten())
    uqi = 4*x.mean()*y.mean()*cov[0, 1]/(cov[0, 0]+cov[1, 1])\
        / (x.mean()**2 + y.mean()**2)
    if uqi == 0:
        uqi = 1e-2
    return mse, 1/uqi
