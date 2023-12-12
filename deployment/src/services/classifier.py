from transformers import AutoModelForSequenceClassification ,AutoTokenizer
from utils.config import setting
import torch.nn.functional as F
import torch 

# Initialize model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(setting.MODEL_PATH).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(setting.TOKENIZER_PATH)

# Round values for a dictionary
def round_dict(input_dict):  
    res_dict = dict()
    for key in input_dict:
        res_dict[key] = round(input_dict[key], 5)
    return res_dict
            
# Preprocess
def preprocess(text):
    model_inputs = tokenizer(text, padding="max_length", max_length=512, return_tensors="pt").to('cuda')
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    if input_ids.shape[1] <= 512:
        print("Get the first part only")
    else:
        print("Combine the first part and last part")
        input_ids = torch.cat((input_ids[:, :256], input_ids[:, -256:]), dim=1)
        attention_mask = torch.cat((attention_mask[:, :256], attention_mask[:, -256:]), dim=1)
    return input_ids, attention_mask

# Classify text and return probabilities
def predict(text, return_all_probabilities):
    input_ids, attention_mask = preprocess(text)
    # Get model scores
    scores = model(input_ids, attention_mask=attention_mask).logits  
    # Apply softmax to get probabilities
    probabilities = F.softmax(scores, dim=1).tolist()[0]  
    # Map probabilities to their corresponding labels
    probability_dict = {label: probability for label, probability in zip(setting.LABEL_LIST, probabilities)}  
    # Sort the dictionary by probability in descending order
    sorted_probability_dict = dict(sorted(probability_dict.items(), key=lambda item: item[1], reverse=True))  
    # If returning all scores
    if return_all_probabilities:  
        # Round the dictionary
        scores_dict = round_dict(sorted_probability_dict)  
        return scores_dict
    # If returning 1 score
    else:  
        # Get the first key of the dictionary
        first_key = list(sorted_probability_dict)[0]  
        # Get the first value of the dictionary
        first_value = round(sorted_probability_dict[first_key], 5)  
        # Round it
        rounded_score = {first_key: first_value}  
        return rounded_score
