import uproot
import pandas as pd
import numpy as np
import sifi_cm.root_aux as raux
import matplotlib.pyplot as plt
from collections import namedtuple
from sifi_cm.data_process import fit_1d, Gaussian_1D
import pickle

from os import path

basepath = path.dirname(__file__)

# from pkg_resources import resource_stream


class HypmedImporter():

    def __init__(self, mapping_file=None) -> None:
        # f = resource_stream(__name__,
        #                     "data/hypmed/preprocess_data_normalized.pkl")
        with open(basepath + "/data/hypmed/preprocess_data_normalized.pkl", "rb") as f:
            self.mapp, self.coord, self.df_pproc = pickle.load(f)

    def get_data(self, filename: str,
                 verbose=False, normalize=True) -> pd.DataFrame:
        with uproot.open(filename) as f:
            for key in f['ordered'].keys():
                if "timestamp" in key.lower():
                    ts = key
                    break
            df = f['ordered'].arrays(library="pd",
                                    expressions=['NeedleNumber',
                                                'PhotonsRoi',
                                                'HVD', ts])
        tot_time_sec = (df[ts].max() - df[ts].min())/10**12
        if verbose:
            print("Total measurement time: ", round(tot_time_sec/60, 2), "[min]")
        df_g = df.groupby("NeedleNumber")["PhotonsRoi"].sum().reset_index()
        if normalize:
            df_g["PhotonsRoi"] /= tot_time_sec
        df_merged = pd.merge(df_g, self.mapp, left_on="NeedleNumber",
                             right_on="index").drop("index", axis=1)
        return df_merged

    @staticmethod
    def get_map(map_file: str = None):
        """Get the mapping for each layer of Hypmed array

        Parameters
        ----------
        map_file : str, optional
            path to the file with mapping from 
            NeedleNumber to position of the hit,
            by default 'data/hypmed/crystal_v2.root'

        Returns
        -------
        (pd.DataFrame, dict)
            Dataframe with each position of needle and
            dictionary with edges coordinates for each layer
        """
        if not map_file:
            # map_file = resource_stream(__name__, "data/hypmed/crystal_v2.root")
            map_file = basepath + "../data/hypmed/crystal_v2.root"
        with uproot.open(map_file) as f:
            mapping = f["tree"].arrays(library="pd")

        coord = {}
        for layer in range(1, 4):
            coord[layer] = raux.Edges(mapping[mapping.layer == layer].x.unique(),
                                      mapping[mapping.layer == layer].y.unique(),
                                      edges=False)
        return mapping, coord

    def process_file(self, fname, data=None):
        if not data:
            data = self.get_data(fname)
        df = pd.merge(
            self.df_pproc, data[["NeedleNumber", "PhotonsRoi"]],
            on="NeedleNumber")
        df["PhotonsRoi_bgfree"] = df["PhotonsRoi"] - df["PhotonsRoi_bg"]
        df["PhotonsRoi_norm"] = df["PhotonsRoi_bgfree"]/df["efficiency_map"]
        return df


def figure_layers(df, coord, label=None, column="PhotonsRoi"):
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(label)
    for layer_num in range(1, 4):
        plt.subplot(130+layer_num)
        plt.hist2d(df[df.layer == layer_num].x, df[df.layer == layer_num].y,
                   bins=[*coord[layer_num].xy_edges()],
                   weights=df[df.layer == layer_num][column], cmap='CMRmap')
        plt.title(f"Layer {layer_num}")
        plt.xlim([coord[1].x[0], coord[1].x[-1]])
        plt.ylim([coord[1].y[0], coord[1].y[-1]])
        fig.supxlabel('x')
        fig.supylabel('y')
        if layer_num == 3:
            cbar = plt.colorbar(label="counts")
        else:
            cbar = plt.colorbar()
        power_range = int(np.log10(df[column].abs().min()))
        cbar.formatter.set_powerlimits((power_range, power_range))
    plt.tight_layout()
    # plt.close()
    return fig


class Reco_image(namedtuple("reco_image", "image edges true_pos")):
    """Class inherited from named tuple.

    Raises
    ------
    ValueError
        If trying to get binWidth when widths are different for X and Y
    """
    def __new__(cls, reco_obj: np.array, reco_edges: raux.Edges,
                norm=True, true_pos=[]):
        if len(reco_obj.shape) == 1:
            if int(np.sqrt(reco_obj.shape[0])) != np.sqrt(reco_obj.shape[0]):
                raise ValueError
            else:
                side_size = int(np.sqrt(reco_obj.shape[0]))
            reco_obj = reco_obj.reshape(
                side_size, side_size).T[::-1, ::-1]
        if norm:
            reco_obj /= reco_obj.sum() * 2 * reco_edges.x_binWidth\
                                       * 2 * reco_edges.y_binWidth
        return super().__new__(cls, reco_obj, reco_edges, true_pos)

    @property
    def projx(self):
        return self.image.sum(axis=0)*2*self.edges.y_binWidth

    @property
    def projy(self):
        return self.image.sum(axis=1)*2*self.edges.x_binWidth

    @property
    def fitx(self):
        return fit_1d(self.edges.x_cent, self.projx)

    @property
    def fity(self):
        return fit_1d(self.edges.y_cent, self.projy)

    def plot(self, label: str = None,
                  fitx=True, fity=True, figsize=(15, 4),
                  correction_x = 0,
                  correction_y = 0):
        """_summary_

        Parameters
        ----------
        label : str, optional
            Figure label, by default None
        """
        xlin = np.linspace(self.edges.x.min(),
                        self.edges.x.max(), 1000)

        fig, axes = plt.subplots(ncols=3, figsize=figsize)
        fig.suptitle(label, y=1)

        p = axes[0].pcolor(self.edges.x + correction_x,
                           self.edges.y + correction_y,
                           self.image, cmap="YlGnBu_r")
        if self.true_pos:
            axes[0].scatter(self.true_pos[0], self.true_pos[1],
                            marker="x", c="r")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].grid(which="both", alpha=0.3)
        plt.colorbar(p, ax=axes[0])

        if self.true_pos:
            for i, x_true in enumerate(self.true_pos[0]):
                axes[1].axvline(x_true, ls="--",
                                c="r", label="true source" if i == 0 else None,
                                zorder=1)
        if fitx:
            axes[1].plot(xlin, Gaussian_1D(xlin, *self.fitx),
                         c="g", label="Gaussian", zorder=1)
            axes[1].set_title(
                f"$\mu_x$ = {round(self.fitx.mean, 2)}[mm],"
                f"$\sigma_x$ = {round(abs(self.fitx.sigma), 2)}")
        axes[1].scatter(self.edges.x_cent + correction_x,
                        self.projx,
                        s=10, label="reco", zorder=2)
        axes[1].set_xlabel("x")
        axes[1].legend()

        if self.true_pos:
            for y_true in self.true_pos[1]:
                axes[2].axvline(y_true, ls="--",
                                c="r", label="true source", zorder=1)
        if fity:
            axes[2].plot(xlin, Gaussian_1D(xlin, *self.fity),
                        c="g", label="Gaussian", zorder=1)
            axes[2].set_title(
                f"$\mu_y$ = {round(self.fity.mean, 2)}[mm],"
                f"$\sigma_y$ = {round(abs(self.fity.sigma), 2)}")
        axes[2].scatter(self.edges.y_cent + correction_y,
                        self.projy,
                        s=10, label="reco", zorder=2)
        axes[2].set_xlabel("y")

        plt.tight_layout()
        return fig


def get_vec_from_layers(df: pd.DataFrame, coord: dict, column: str):
    histo_data = [plt.hist2d(df[df.layer == layer_num].x,
                             df[df.layer == layer_num].y,
                             bins=[*coord[layer_num].xy_edges()],
                             weights=df[df.layer == layer_num][column],
                             cmap='CMRmap')
                  for layer_num in range(1, 4)]
    plt.close()
    return np.hstack([dep[0][::-1, ::-1].flatten()
                      for dep in histo_data[::-1]])
