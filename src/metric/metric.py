import gc
import tqdm
import torch
import numpy as np
from typing import Tuple

from nltk.translate.bleu_score import sentence_bleu
from tqdm.auto import trange

from wieting_similarity.similarity_evaluator import SimilarityEvaluator


from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def classify_preds(preds, device, batch_size, soft=False, threshold=0.8):
    print('Calculating style of predictions')
    results = []

    model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    for i in tqdm.tqdm(range(0, len(preds), batch_size)):
        batch = tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
        batch = {key: value.to(device) for key, value in batch.items()}  # Move the batch to the same device as the model
        with torch.inference_mode():
            logits = model(**batch).logits
        if soft:
            result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        else:
            result = (logits[:, 1] > threshold).cpu().numpy()
        results.extend([1 - item for item in result])
    return results



def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1
        
    return float(bleu_sim / counter)


def wieting_sim(inputs, preds, batch_size):
    assert len(inputs) == len(preds)
    print('Calculating similarity by Wieting subword-embedding SIM model')

    sim_evaluator = SimilarityEvaluator()
    
    sim_scores = []
    
    for i in tqdm.tqdm(range(0, len(inputs), batch_size)):
        sim_scores.extend(
            sim_evaluator.find_similarity(inputs[i:i + batch_size], preds[i:i + batch_size])
        )
        
    return np.array(sim_scores)


def do_cola_eval(preds, device, batch_size):
    print('Calculating CoLA acceptability stats')

    model_name = "cointegrated/roberta-large-cola-krishna2020"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    outputs = []

    # Process predictions in batches
    for i in tqdm.tqdm(range(0, len(preds), batch_size)):
        batch_preds = preds[i:i+batch_size]  # Get a batch of predictions

        # Tokenize the batch of predictions
        batch_input_ids = tokenizer.batch_encode_plus(
            batch_preds, add_special_tokens=True, return_tensors="pt", padding=True
        )

        batch_input_ids = {key: value.to(device) for key, value in batch_input_ids.items()}

        with torch.no_grad():
            # Ensure the batch size is not 0 (e.g., for the last batch)
            if batch_input_ids["input_ids"].shape[0] > 0:
                batch_logits = model(batch_input_ids["input_ids"]).logits
                batch_outputs = torch.argmax(batch_logits, dim=1).tolist()
                outputs.extend(batch_outputs)

    return np.array(outputs)


def calculate_metric(inputs: list, preds: list, batch_size=32) -> Tuple[float, float, float, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'running on {device}')

    # accuracy of style transfer
    accuracy_by_sent = classify_preds(preds, device, batch_size)
    accuracy = sum(accuracy_by_sent)/len(preds)
    cleanup()
    
    # similarity
    bleu = calc_bleu(inputs, preds)
    
    similarity_by_sent = wieting_sim(inputs, preds, batch_size)
    avg_sim_by_sent = similarity_by_sent.mean()
    cleanup()
    
    # fluency
    cola_stats = do_cola_eval(preds, device, batch_size)
    cola_acc = sum(cola_stats) / len(preds)
    cleanup()
    
    # count metrics
    joint = sum(accuracy_by_sent * similarity_by_sent * cola_stats) / len(preds)
    
    # write res to table
    print('| ACC | SIM |  FL  |   J   | BLEU |\n')
    print('| --- | --- | ---- |  ---  | ---- |\n')
    print(f'|{accuracy:.4f}|{avg_sim_by_sent:.4f}|{cola_acc:.4f}|{joint:.4f}|{bleu:.4f}|\n')
    return accuracy, avg_sim_by_sent, cola_acc, joint, bleu