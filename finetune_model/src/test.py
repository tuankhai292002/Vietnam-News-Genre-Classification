import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import pandas as pd
import hydra
import os
import torch
from transformers import AutoTokenizer , DataCollatorWithPadding
from omegaconf import DictConfig 

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import set_seed
import evaluate
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

evaluate_data =load_dataset(
        "csv",
        data_files="/data/khaitt4/tuankhai_data_moderation/data_moderation/dataset/dataset/undersampled_data/small_data/phanchinh/test_new.csv",
        cache_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
        split='train'
    )
tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI/models--vietdata--vietnamese-content-cls/snapshots/64fd7183ce864557dc77900071a680cdcc0a652f",
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
tokenized_data_eval = evaluate_data.map(preprocess_function, batched=True).shuffle(seed=41) #data để test
training_args = TrainingArguments( #Tham số để train   
        output_dir="/data/khaitt4/tuankhai_data_moderation/.cache/TuanKhai_Intern_AI",
        per_device_eval_batch_size=100,
        per_device_train_batch_size=100,
    ) 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(f"label: {labels}")
    print(f"preds: {preds}")
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    print(f"f1 {f1}")
    print(f"recall {recall}")
    print(f"precision {precision}")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
model_name = "/data/khaitt4/tuankhai_data_moderation/.cache/model_logs/models/checkpoint-19590"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda")
model = model.to(device)
trainer = Trainer(
        model=model,
        eval_dataset=tokenized_data_eval,
        compute_metrics=compute_metrics,
    )    



    # trainer.train(resume_from_checkpoint="/data/khaitt4/tuankhai_data_moderation/.cache/model_logs/models/checkpoint-6365")
result=trainer.evaluate(eval_dataset=tokenized_data_eval)
print(result)
