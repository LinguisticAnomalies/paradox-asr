'''
Fine-tune BERT with sequential classification head
'''


import logging
import configparser
import logging
import gc
import sys
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from datasets import load_metric
from transformers import TrainingArguments, Trainer
import pandas as pd


CONTEXT_LENGTH = 512
RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS = 10


def tokenize_function(element):
    """
    batchfy the tokenization
    :param element: a frament of dataset
    :type element: transformers.Dataset
    """
    return tokenizer(
        element["trans"],
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=True,
        max_length=CONTEXT_LENGTH,
        padding=True)


def prepare_dataset(file_path, token=True):
    """
    tokenize the dataset
    :param file_path: the location to the file
    :type file_path: str
    """
    dt = pd.read_csv(file_path)
    # merge to transcript level
    # NOTE: this step is required if using ppl-audio data.
    # as it was segmented by sentence for ASR
    # NOTE: for this step, only file, tran, label are needed
    dt = dt[["file", "trans", "label"]]
    dt = dt.groupby(["file", "label"])["trans"].apply(". ".join).reset_index()
    dt["trans"] = dt["trans"].str.lower()
    # change label to int
    dt["label"] = dt["label"].astype(int)
    if token:
        dt = Dataset.from_pandas(dt)
        dt = dt.map(tokenize_function, batched=True)
    return dt


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
    pred_labels = []
    files = []
    res_df = pd.DataFrame()
    for _, row in eval_df.iterrows():
        text = row["trans"]
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
        predicted_label = logits.argmax().item()
        pred_labels.append(predicted_label)
    res_df["file"] = files
    res_df["bert_label"] = pred_labels
    return res_df


def compute_metrics(eval_pred):
    """
    compute accuracy for the fine-tuned BERT model

    :param eval_pred: _description_
    :type eval_pred: _type_
    :return: _description_
    :rtype: _type_
    """
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    # if don't have a fine-tuned BERT model
    if not os.path.exists(os.path.join(config["PATH"]["PrefixModel"], "bert")):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized_train = prepare_dataset(
            os.path.join(config["PATH"]["PrefixManifest"], "adr_train.csv"))
        tokenized_test = prepare_dataset(
            os.path.join(config["PATH"]["PrefixManifest"], "adr_test.csv"))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2)
        output_log = f"../logs/fine-tune-bert.log"
        with open(output_log, "w") as log_file:
            sys.stdout = log_file
            logging.basicConfig(
                format='%(asctime)s : %(levelname)s : %(message)s',
                filemode="a", level=logging.INFO,
                filename=output_log)
            training_args = TrainingArguments(
                output_dir="../outputs/",
                num_train_epochs=EPOCHS,
                warmup_steps=100,
                fp16=True,
                learning_rate=1e-4,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                evaluation_strategy='epoch',
                save_strategy="epoch",
                logging_strategy="epoch",
                # prediction_loss_only=True,
                do_train=True,
                do_eval=True,
                max_grad_norm=1.0,
                seed=RANDOM_SEED,
                data_seed=RANDOM_SEED,
                save_total_limit=2,
                load_best_model_at_end=True,
                report_to="none",
                metric_for_best_model="accuracy",
                greater_is_better=True
            )
            if training_args.do_train:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_test,
                    compute_metrics=compute_metrics
                )
                trainer.train()
            model.save_pretrained("../ft-models/bert/")
            del trainer, model
            gc.collect()
    # get BERT prediction label
    else:
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(config["PATH"]["PrefixModel"], "bert"), num_labels=2)
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        adr_test = prepare_dataset(
            os.path.join(config["PATH"]["PrefixManifest"], "adr_test.csv"), token=False)
        bert_res = evaluate_bert_sequence(adr_test, bert_model, bert_tokenizer)
        bert_res.to_csv(
            "../followup/bert_verbatim_pred.csv", index=False)
