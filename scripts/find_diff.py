'''
Find the difference in prediction for error analysis
'''


import configparser
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd


def prepare_dataset(file_path, tran_type):
    """

    :param file_path: the local path to the dataset
    :type file_path: str
    :param tran_type: the type of the transcripts for merging
    :type tran_type: str
    :return: the evaluaiton dataframe
    :rtype: pd.DataFrame
    """
    dt = pd.read_csv(file_path)
    # change label to int
    dt["label"] = dt["label"].astype(int)
    # replace nan into empty string
    dt = dt.replace(np.nan, " ", regex=True)
    if tran_type == "verbatim":
        dt = dt[["file", "trans", "label"]]
        dt = dt.groupby(["file", "label"])["trans"].apply(". ".join).reset_index()
        dt["trans"] = dt["trans"].str.lower()
        dt.columns = ["file", "true_label", "verbatim_trans"]
    elif tran_type == "asr":
        dt = dt[["file", "text", "label"]]
        dt = dt.groupby(["file", "label"])["text"].apply(". ".join).reset_index()
        dt["text"] = dt["text"].str.lower()
        dt.columns = ["file", "true_label", "asr_trans"]
    else:
        raise ValueError("Wrong transcript type")
    return dt


if __name__ == "__main__":
    start_time = datetime.now()
    config = configparser.ConfigParser()
    config.read("config.ini")
    model_cards = [
        "wav2vec2-base-960h",
        "wav2vec2-large-960h",
        "wav2vec2-large-960h-lv60",
        "wav2vec2-large-960h-lv60-self",
        "hubert-large-ls960-ft"]
    modes = ["ft", "ori"]
    adr_test = prepare_dataset(
        os.path.join(config["PATH"]["PrefixManifest"], "adr_test.csv"), "verbatim")
    adr_train = prepare_dataset(
        os.path.join(config["PATH"]["PrefixManifest"], "adr_train.csv"), "verbatim")
    pitt_meta = pd.read_csv(
        os.path.join(config["PATH"]["PrefixManifest"], "pitt_merged.tsv"), sep="\t")
    pitt_meta = pitt_meta[["file", "mmse"]]
    verbatim_pred = pd.read_csv("../followup/bert_verbatim_pred.csv")
    print(verbatim_pred.head())
    for model_card in model_cards:
        for mode in modes:
            # ASR generated transcripts from ADReSS training set
            adr_train_trans = prepare_dataset(
                os.path.join(config["PATH"]["PrefixTrans"], f"{mode}-{model_card}-adr_train.csv"),
                "asr")
            adr_train_trans = adr_train_trans.merge(adr_train, on=["file", "true_label"])
            adr_train_trans = adr_train_trans.merge(pitt_meta, on="file")
            adr_train_trans = adr_train_trans[["file", "true_label",
                                               "verbatim_trans", "asr_trans", "mmse"]]
            adr_train_trans.to_csv(
                f"../followup/{mode}-{model_card}-adr_train-followup.csv", index=False)
            # ASR generated transcripts from ADReSS test set
            sys.stdout.write(f"merging {mode} {model_card} for error analysis\n")
            adr_test_trans = prepare_dataset(
                os.path.join(config["PATH"]["PrefixTrans"], f"{mode}-{model_card}-adress.csv"),
                "asr")
            # add verbatim transcripts
            full_df = adr_test.merge(adr_test_trans, on=["file", "true_label"])
            # add mmse
            full_df = full_df.merge(pitt_meta, on="file", how="inner")
            # add verbatim bert predicted label
            full_df = full_df.merge(verbatim_pred, on="file", how="inner")
            # add asr bert predicted label
            asr_pred = pd.read_csv(
                f"../pred/{mode}-{model_card}-pred.csv")
            asr_pred.columns = ["file", "asr_label"]
            full_df = full_df.merge(asr_pred, on="file")
            full_df = full_df[["file", "true_label", "bert_label", "asr_label",
                               "verbatim_trans", "asr_trans", "mmse"]]
            full_df.to_csv(
                f"../followup/{mode}-{model_card}-adress-followup.csv", index=False)
    sys.stdout.write(f"Total time: {datetime.now()-start_time}\n")