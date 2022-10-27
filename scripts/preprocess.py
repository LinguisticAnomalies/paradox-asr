'''
Various functions for preprocessing, including:
    - add metadata for utterance level transcripts
    - generate manifest on vairous subset
Before running this script, please make sure you have the audio recordings resampled and segmented
'''


import os
import pandas as pd


def add_loc(cha_df_path, audio_prefix):
    """
    add audio segment location to the TRESTLE output
    :param cha_df: path to TRESTLE output dataframe
    :type cha_df: str
    :param audio_prefix: the prefix to audio segments
    :type audio_prefix: str
    """
    cha_df = pd.read_csv(cha_df_path)
    cha_df["trans"] = cha_df["trans"].str.upper()
    locs = []
    for _, row in cha_df.iterrows():
        sub_file = f"{row['file']}-{row['index']}.wav"
        loc = os.path.join(audio_prefix, sub_file)
        locs.append(loc)
    cha_df["locs"] = locs
    cha_df.to_csv(cha_df_path, index=False)


def generate_subset():
    """
    generate subsets
    """
    db_con = pd.read_csv("/edata/lixx3013/manifest/db_con_cha.csv")
    db_dem = pd.read_csv("/edata/lixx3013/manifest/db_dem_cha.csv")
    db_full = pd.concat([db_con, db_dem])
    db_full.to_csv("/edata/lixx3013/manifest/db_full.csv", index=False)
    pitt = pd.read_csv("/edata/lixx3013/manifest/pitt_merged.tsv", sep="\t")
    pitt_meta = pitt[["file", "dx", "inADReSS", "ADReSS_train","ADReSS_test","label"]]
    pitt_label = pitt_meta[["file", "label"]]
    # full ADReSS with labels
    adr_ids = pitt_meta.loc[pitt_meta["inADReSS"] == 1]["file"].values.tolist()
    adr_subset = db_full.loc[db_full["file"].isin(adr_ids)]
    adr_subset = adr_subset.merge(pitt_label, on="file", how="inner")
    adr_subset["label"] = adr_subset["label"].astype(int)
    adr_subset.to_csv("/edata/lixx3013/manifest/adr_full.csv", index=False)
    # ADReSS training set
    adr_train_ids = pitt_meta.loc[pitt_meta["ADReSS_train"] == 1]["file"].values.tolist()
    adr_train = db_full.loc[db_full["file"].isin(adr_train_ids)]
    adr_train = adr_train.merge(pitt_label, on="file", how="inner")
    adr_train["label"] = adr_train["label"].astype(int)
    adr_train.to_csv("/edata/lixx3013/manifest/adr_train.csv", index=False)
    # ADReSS test set
    adr_test_ids = pitt_meta.loc[pitt_meta["ADReSS_test"] == 1]["file"].values.tolist()
    adr_test = db_full.loc[db_full["file"].isin(adr_test_ids)]
    adr_test = adr_test.merge(pitt_label, on="file", how="inner")
    adr_test["label"] = adr_test["label"].astype(int)
    adr_test.to_csv("/edata/lixx3013/manifest/adr_test.csv", index=False)
    # ADReSS controls
    adr_con_ids = pitt_meta.loc[(pitt_meta["inADReSS"] == 1) & (pitt_meta["label"]==0)]["file"].values.tolist()
    adr_con = db_full.loc[db_full["file"].isin(adr_con_ids)]
    adr_con = adr_con.merge(pitt_label, on="file", how="inner")
    adr_con["label"] = adr_con["label"].astype(int)
    adr_con.to_csv("/edata/lixx3013/manifest/adr_con.csv", index=False)
    # ADReSS dementia
    adr_dem_ids = pitt_meta.loc[(pitt_meta["inADReSS"] == 1) & (pitt_meta["label"]==1)]["file"].values.tolist()
    adr_dem = db_full.loc[db_full["file"].isin(adr_dem_ids)]
    adr_dem = adr_dem.merge(pitt_label, on="file", how="inner")
    adr_dem["label"] = adr_dem["label"].astype(int)
    adr_dem.to_csv("/edata/lixx3013/manifest/adr_dem.csv", index=False)
    # exadr
    exadr_ids = pitt_meta.loc[pitt_meta["inADReSS"] == 0]["file"].values.tolist()
    # ProbableAD and Control only
    pitt_exadr_dem = pitt.loc[pitt["dx"] == "ProbableAD"]["file"].values.tolist()
    exadr_dem = db_full.loc[db_full["file"].isin(pitt_exadr_dem)]
    exadr_dem["label"] = 1
    exadr_dem.to_csv("/edata/lixx3013/manifest/exadr_dem.csv", index=False)
    pitt_exadr_con = pitt.loc[pitt["dx"] == "Control"]["file"].values.tolist()
    exadr_con = db_full.loc[db_full["file"].isin(pitt_exadr_con)]
    exadr_con["label"] = 0
    exadr_con.to_csv("/edata/lixx3013/manifest/exadr_con.csv", index=False)


if __name__ == "__main__":
    # add_loc("/edata/lixx3013/manifest/db_con_cha.csv",
    #         "/edata/lixx3013/audio-pieces/pitt/control")
    # add_loc("/edata/lixx3013/manifest/db_dem_cha.csv",
    #         "/edata/lixx3013/audio-pieces/pitt/dementia")
    # add_loc("/edata/lixx3013/manifest/wls_00_cha.csv",
    #         "/edata/lixx3013/audio-pieces/wls/00")
    # add_loc("/edata/lixx3013/manifest/wls_01_cha.csv",
    #         "/edata/lixx3013/audio-pieces/wls/01")
    # add_loc("/edata/lixx3013/manifest/wls_02_cha.csv",
    #         "/edata/lixx3013/audio-pieces/wls/02")
    generate_subset()
