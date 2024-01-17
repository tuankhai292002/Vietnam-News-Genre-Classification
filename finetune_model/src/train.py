import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import pandas as pd
import hydra
import os
import torch
from transformers import AutoTokenizer 
from omegaconf import DictConfig 
from handle_data import *
from utils import *
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import set_seed
import evaluate
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
import torch

torch.cuda.empty_cache()
os.environ['TRANSFORMERS_CACHE'] = "/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI"
os.environ['HF_DATASETS_CACHE'] = "/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI"

@hydra.main(config_path='../conf', config_name='config')
def training(cfg : DictConfig) -> None: 
    id2label, label2id = from_labels_to_ids(cfg.data.labels) 
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_information.embedding_model,
        num_labels = 13,
        token = "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",
        cache_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
        ignore_mismatched_sizes=True        ,
    )
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True , 
            padding="max_length",
            add_special_tokens=False,
            max_length= 256
            # cfg.hyperparameter.max_lenght_token
        )
    set_seed(8)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model_information.prediction_model,
        # pretrained_model_name_or_path = "/data/khaitt4/tuankhai_data_moderation/.cache/model_logs/models/checkpoint-37080",
        num_labels = len(cfg.data.labels), 
        id2label = id2label, 
        label2id = label2id,
        ignore_mismatched_sizes=True,
        cache_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
    ) 
    device = torch.device("cuda")
    model = model.to(device)
    
    train_data = load_dataset(
        "csv",
        data_files=cfg.data.train_path,
        cache_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
        split='train'
    )

    evaluate_data =load_dataset(
        "csv",
        data_files=cfg.data.evaluate_path,
        cache_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
        split='train'
    )

    tokenized_data_train = train_data.map(preprocess_function, batched=True).shuffle(seed=41) #data để train
    tokenized_data_eval = evaluate_data.map(preprocess_function, batched=True).shuffle(seed=41) #data để test

    training_args = TrainingArguments( 
        output_dir=cfg.model_information.output_dir,
        logging_dir=cfg.model_information.log_dir,
        logging_strategy='steps', 
        logging_steps=8,
        num_train_epochs=cfg.hyperparameter.num_epochs,
        per_device_eval_batch_size=cfg.hyperparameter.batch_size_eval,
        per_device_train_batch_size=cfg.hyperparameter.batch_size_train,
        learning_rate=cfg.hyperparameter.learning_rate,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        weight_decay=cfg.hyperparameter.weight_decay,
    ) 
 
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data_train,
        eval_dataset=tokenized_data_eval,
        compute_metrics=compute_metrics,
    )    

    trainer.train()

if __name__ == "__main__":
    training()
