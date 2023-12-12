import numpy as np
import evaluate

def from_labels_to_ids(labels):
    id2label = dict()
    label2id = dict()
    for label,i in zip(labels, range(0,len(labels))):
        id2label[i] = label
        label2id[label] = i
    
    return id2label, label2id

def _compute_metrics(eval_pred):
    accuracy = evaluate('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions , axis=1)
    return accuracy.compute(
        predictions=predictions,
        references=labels
    )