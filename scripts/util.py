'''
This script includes several utility functions for this project
'''


import sys
import re
import pandas as pd
import numpy as np
from evaluate import load
from transformers import Wav2Vec2Processor, AutoModelForCTC


def clean_text_asr(line):
    """
    clean the extracted sentence and return it
    :param line: the raw sentence extracted from .cha file
    :type line: str
    """
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    line = re.sub(chars_to_ignore_regex, "", line)
    line = line.upper()
    return line


def read_trans(mode, model_card, data, group=None):
    """
    read ASR-generated transcripts,
    force NA predictions as empty strings,
    return the dataframe

    :param model_card: the model used for generating transcripts
    :type model_card: str
    :param mode: the choice of the model status, i.e., fine-tuned or pre-trained
    :type mode: str
    :param dataset: the specific dataset to subset, i.e., adr_train or adress (test set)
    :type dataset: str
    :param group: the subset of the dataset, i.e., con or dem
    :type group: str, default is None
    return: ASR-generated transcripts as dataframe
    rtype: pd.DataFrame
    """
    if group:
        eval_df = pd.read_csv(
            f"../asr-trans/{mode}-{model_card}-{data}-{group}.csv")
    else:
        eval_df = pd.read_csv(
            f"../asr-trans/{mode}-{model_card}-{data}.csv")
    eval_df = eval_df.replace(np.nan, " ", regex=True)
    return eval_df


def prepare_dataset(file_path, trans_type):
    """

    :param file_path: the local path to the dataset
    :type file_path: str
    :return: the evaluaiton dataframe
    :rtype: pd.DataFrame
    """
    dt = pd.read_csv(file_path)
    # change label to int
    dt["label"] = dt["label"].astype(int)
    # replace nan into empty string
    dt = dt.replace(np.nan, " ", regex=True)
    # merge to transcirpt level
    if trans_type == "verbatim":
        dt = dt[["file", "trans", "label"]]
        dt = dt.groupby(["file", "label"])["trans"].apply(". ".join).reset_index()
        dt["trans"] = dt["trans"].str.lower()
    elif trans_type == "asr":
        dt = dt[["file", "text", "label"]]
        dt = dt.groupby(["file", "label"])["text"].apply(". ".join).reset_index()
        dt["text"] = dt["text"].str.lower()
    else:
        raise ValueError("Wrong transcript type")
    dt = dt.sample(frac=1)
    return dt


def get_wcer(trans_df):
    """
    get WER and CER for ASR-generated transcripts

    :param trans_df: the dataframe of ASR-generated transcripts
    :type trans_df: pd.DataFrame
    """
    wer_metric = load("wer")
    cer_metric = load("cer")
    wer_score = wer_metric.compute(
        predictions=trans_df["text"].values.tolist(),
        references=trans_df["trans"].values.tolist())
    cer_score = cer_metric.compute(
        predictions=trans_df["text"].values.tolist(),
        references=trans_df["trans"].values.tolist())
    sys.stdout.write(
        f"\tWER: {round(wer_score, 3)}, CER: {round(cer_score, 3)}\n")


def get_wcer_per_trans(trans_df):
    """
    get WERC/CER per pariticipant with given dataset

    :param trans_df: participant-level transcripts
    :type trans_df: pd.DataFrame
    """
    wers = []
    cers = []
    for _, row in trans_df.iterrows():
        wer_metric = load("wer")
        cer_metric = load("cer")
        pred = row["text"]
        refs = row["trans"]
        wer_metric.add(prediction=pred, reference=refs)
        cer_metric.add(prediction=pred, reference=refs)
        wers.append(round(wer_metric.compute(), 3))
        cers.append(round(cer_metric.compute(), 3))
    trans_df["wer"] = wers
    trans_df["cer"] = cers
    return trans_df



def load_model(model_card, fine_tune_indicator):
    """
    load the model and corresponding processor with model card name

    :param model_card: the name of model card from huggingface
    :type model_card: str
    :param fine_tune_indicator: if loading fine-tuned model
    :type fine_tune_indicator: bool
    :return: model and processor
    :rtype: transformers.AutoModelForCTC, transformers.Wav2Vec2Processor
    """
    if fine_tune_indicator:
        model = AutoModelForCTC.from_pretrained(
            f"../ft-models/{model_card}")
    else:
        model = AutoModelForCTC.from_pretrained(f"facebook/{model_card}")
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/{model_card}")
    return model, processor
