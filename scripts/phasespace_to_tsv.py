import uproot
from tqdm import tqdm 
import argparse
import os
import pandas as pd

COLUMNS = ["ParticleID", "ParticleEnergy", "ParticlePositionfX",
           "ParticlePositionfY", "ParticlePositionfZ", "ParticleMomentumfX",
           "ParticleMomentumfY", "ParticleMomentumfZ"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", help="Extract data from PhaseSpace files",
                        nargs="*")
    parser.add_argument("--job_num", help="Number of parralel job",
                        default=1, type=int)
    args = parser.parse_args()

    filenames = [f for f in args.filenames if f.endswith(".root")]

    for filename in tqdm(filenames, desc=str(args.job_num)):
    # for filename in filenames:
        df = pd.DataFrame()
        # print(filename)
        # print(f"{i+1} out of {len(args.filenames)}")
        output = filename.replace(".root", "_extracted.tsv")
        if os.path.isfile(output):
            continue
        # open(output, 'w').close()
        with uproot.open(filename) as file:
            sec = file["Secondaries;1"]
            try:
                df = sec.arrays(library="pd")
            except:
                print("Did not extracted")
                print(filename, "error")
                continue
        if isinstance(df, pd.DataFrame):
            df = df.reset_index().drop(["entry", "subentry"], axis=1)[COLUMNS]
        else:
            print("SOMETHING IS WRONG with", filename)
            print(filename, "error")
            continue
        df.to_csv(output, header=False, index=False, sep="\t")
        print(filename, "success")
            # print(df.columns)
            # gamm_energy = []
            # gamm_pos = []
            # gamm_dir = []
            # for batch in sec.iterate(filter_name="/Particle(ID|Energy|Position|Momentum)/",
            #                               step_size="5 MB", entry_stop=None,
            #                               library="pd"):
                # batch[batch["ParticleID"] == 22].reset_index()\
                #                                 .drop(["entry", "subentry"], axis=1)\
                #                                 .to_csv(output, mode='a', header=False,
                #                                         index=False, sep="\t")

