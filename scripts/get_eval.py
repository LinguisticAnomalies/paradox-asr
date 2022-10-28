'''
Use fine-tuned BERT model
'''


import configparser
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_curve, auc, accuracy_score
from util import prepare_dataset

CONTEXT_LENGTH = 512

def evaluate_bert_sequence(eval_df, model, tokenizer):
    """
    row-by-row evaluation for BERT, with ASR generated transcripts,
    return accuracy, auc, lower ci and upper ci
    :param eval_df: the dataframe for evaluation
    :type eval_df: pd.DataFrame
    :param model: the fine-tuned bert model
    :type model: transformers.AutoModelForSequenceClassification
    :param tokenizer: the associated tokenizer
    :type: transformers.AutoTokenizer
    """
    model.to("cpu")
    pred_labels = []
    probs = []
    files = []
    pred_df = pd.DataFrame()
    for _, row in eval_df.iterrows():
        text = row["text"]
        files.append(row["file"])
        encoded_dict = tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=CONTEXT_LENGTH,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True)
        with torch.no_grad():
            logits = model(**encoded_dict).logits
            # prossiblity for AUC
            prob = F.softmax(logits, dim=-1).tolist()[0][1]
            probs.append(prob)
        predicted_label = logits.argmax().item()
        pred_labels.append(predicted_label)
    true_labels = eval_df["label"].values.tolist()
    # calculate accuracy
    eval_acc = accuracy_score(true_labels, pred_labels)
    # calculate auc
    fpr, tpr, _, = roc_curve(true_labels, probs)
    eval_auc_level = auc(fpr, tpr)
    pred_df["file"] = files
    pred_df["pred"] = pred_labels
    return round(eval_acc, 3), round(eval_auc_level, 3), pred_df


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
    modes = ["ori", "ft"]
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(config["PATH"]["PrefixModel"], "bert"), num_labels=2)
    output_log = f"../logs/eval.log"
    with open(output_log, "w") as log_file:
        sys.stdout = log_file
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            filemode="w", level=logging.INFO,
            filename=output_log)
        for model_card in model_cards:
            for mode in modes:
                test_path = os.path.join(
                    config["PATH"]["PrefixTrans"], f"{mode}-{model_card}-adress.csv")
                test_df = prepare_dataset(test_path, "asr")
                acc, auc_level, pred_df = evaluate_bert_sequence(
                    test_df, bert_model, bert_tokenizer)
                sys.stdout.write(f"{mode} {model_card} acc: {acc}, auc: {auc_level}\n")
                pred_df.to_csv(f"../pred/{mode}-{model_card}-pred.csv", index=False)
        sys.stdout.write(f"Total time: {datetime.now()-start_time}\n")
