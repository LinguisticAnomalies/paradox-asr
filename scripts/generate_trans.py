'''
Generate transcripts from ASR models, including:
    - wav2vec2-base-960h
    - wav2vec2-large-960h-lv60-self
    - wav2vec2-large-960h-lv60
    - wav2vec2-large-960h
    - hubert-large-ls960-ft
'''


import argparse
import sys
import gc
from tqdm import tqdm
from datetime import datetime
import torch
import soundfile as sf
import pandas as pd
from evaluate import load
from util import load_model


torch.manual_seed("55414")


def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine_tune", action="store_true",
        help="""if using the fine-tuned model"""
    )
    parser.add_argument(
        "--no_fine_tune", action="store_false",
        help="""if NOT using the fine-tuned model"""
    )
    parser.add_argument(
        "--dataset", type=str,
        help="""the evaluation model"""
    )
    return parser.parse_args()


def transcribe_eval(model, processor, eval_df, df_type):
    """
    transcribe sentence-level audio into text

    :param model: the ASR model
    :type model: transformers.AutoModelForCTC
    :param processor: the ASR processor
    :type processor: transformers.Wav2Vec2Processor
    :param eval_df: the dataframe for evaluation
    :type eval_df: pd.DataFrame
    :return: the dataframe with ASR transcribed text and verbatim
    :rtype: pd.DataFrame
    """
    wer_metric = load("wer")
    cer_metric = load("cer")
    res_df = pd.DataFrame()
    file = []
    # original text
    trans = []
    text = []
    label = []
    locs = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        try:
            audio_input, sample_rate = sf.read(row["locs"])
            inputs = processor(
                audio_input, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            pred_tran = transcription[0]
            file.append(row["file"])
            trans.append(row["trans"])
            # ASR generated transcripts
            text.append(pred_tran)
            locs.append(row["locs"])
            if df_type == "wls":
                continue
            else:
                label.append(row["label"])
            del inputs
            gc.collect()
        except RuntimeError:
            continue
    res_df["file"] = file
    res_df["trans"] = trans
    res_df["text"] = text
    res_df["locs"] = locs
    if df_type == "wls":
        pass
    else:
        res_df["label"] = label
    wer_score = wer_metric.compute(
        predictions=text, references=trans)
    cer_score = cer_metric.compute(
        predictions=text, references=trans)
    sys.stdout.write(f"WER: {round(wer_score, 3)}, CER: {round(cer_score, 3)}\n")
    return res_df


if __name__ == "__main__":
    start_time = datetime.now()
    args = parge_args()
    if args.fine_tune:
        ft_ind = True
        ft_holder = "ft"
    else:
        ft_ind = False
        ft_holder = "ori"
    model_cards = [
        "wav2vec2-base-960h",
        "wav2vec2-large-960h",
        "wav2vec2-large-960h-lv60",
        "wav2vec2-large-960h-lv60-self",
        "hubert-large-ls960-ft"]
    for model_card in model_cards:
        model, processor = load_model(model_card, ft_ind)
        if args.dataset == "adress":
            eval_df = pd.read_csv(
                "/edata/lixx3013/manifest/adr_test.csv")
        elif args.dataset == "wls":
            eval_df = pd.read_csv(
                "/edata/lixx3013/manifest/wls_02_cha.csv")
        elif args.dataset == "adr_train":
            eval_df = pd.read_csv(
                "/edata/lixx3013/manifest/adr_train.csv")
        out_file = f"../asr-trans/{ft_holder}-{model_card}-{args.dataset}.csv"
        sys.stdout.write(f"Evaluate the {ft_holder} {model_card} on {args.dataset}...\n")
        res_df = transcribe_eval(model, processor, eval_df, args.dataset)
        res_df.to_csv(out_file, index=False)
        del model, processor
        gc.collect()
    sys.stdout.write(f"Total running time: {datetime.now() - start_time}\n")