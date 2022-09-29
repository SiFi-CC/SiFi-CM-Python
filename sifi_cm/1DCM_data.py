import root_aux as raux
import numpy as np
import uproot
from collections import namedtuple


class meas_data(namedtuple("meas_data",
                           "energy, counts")):
    @property
    def energy_plane(self):
        return self.energy[:32] + self.energy[32:]

    @property
    def counts_plane(self):
        return self.counts[:32] + self.counts[32:]

    def __add__(self, other):
        return meas_data(self.energy + other.energy, self.counts + other.counts)

    def __sub__(self, other):
        energy_tmp = self.energy - other.energy
        energy_tmp[energy_tmp < 0] = 0
        counts_tmp = self.counts - other.counts
        counts_tmp[counts_tmp < 0] = 0
        return meas_data(energy_tmp, counts_tmp)

    def __truediv__(self, other):
        return meas_data(self.energy / other.energy, self.counts / other.counts)

    def normalize(self, en_scale, c_scale):
        return meas_data(self.energy/en_scale, self.counts/c_scale)


def get_data(dir_name, thres_low=0.0, thres_up=np.inf):
    with uproot.open("./" + dir_name + "/fiberCoincidences_calib.root") as file:
        df = file["calibratedData;1"].arrays(
            ["ScaFiberNumber", "ScaE", "ScaTimeStampR", "ScaTimeStampL"], library="pd")
    tot_time_sec = (df[["ScaTimeStampR", "ScaTimeStampL"]].max().max()
                    - df[["ScaTimeStampR", "ScaTimeStampL"]].min().min())/10**12
    df = df.drop(df[(df["ScaE"] < thres_low) | (df["ScaE"] > thres_up)].index)
    energy = df.groupby("ScaFiberNumber").sum()["ScaE"].values/tot_time_sec
    counts = df.groupby("ScaFiberNumber").count()["ScaE"].values/tot_time_sec
    return meas_data(energy, counts), df

if __name__ == "__main__":
    bg_data, _ = get_data("2021-12-17_-_14-38-16_-_Test1DCMNoSource_15min")
    nomask_data, _ = get_data("2021-12-17_-_13-02-26_-_Test1DCMNoMask7_7_5min")
    nomask_bgfree_data = nomask_data - bg_data
    nomask_bgfree_scaled_data = nomask_bgfree_data.normalize(
        max(nomask_bgfree_data.energy), max(nomask_bgfree_data.counts))

    hmat = raux.get_hmat(
        "../../Python_reco/input/1d_simulation/matr225_170_n1e6_det32_2lay_nowallpetcut31_mask467_70mm_1d_shifted.root")
    edges = raux.get_source_edges(
        "../../Python_reco/input/1d_simulation/matr225_170_n1e6_det32_2lay_nowallpetcut31_mask467_70mm_1d_shifted.root")

    xlin = np.linspace(-32, 32, 1000)
