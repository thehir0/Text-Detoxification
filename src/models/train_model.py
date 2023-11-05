import torch
from tqdm import tqdm
import warnings
import sys
import argparse
from datasets import Dataset
from src.data.make_dataset import FilteredDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
sys.path.insert(0, '..')
from src.metric.metric import calculate_metric

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataset():
    train_dataset = FilteredDataset(test=False)
    test_dataset = FilteredDataset(test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    train_data = {'text': [], 'label': []}
    test_data = {'text': [], 'label': []}

    for batch, labels in tqdm(train_dataloader):
        batch = list(batch)
        labels = list(labels)
        train_data['text'].extend(batch)
        train_data['label'].extend(labels)

    for batch, labels in tqdm(test_dataloader):
        batch = list(batch)
        labels = list(labels)
        test_data['text'].extend(batch)
        test_data['label'].extend(labels)

    train_dataset = Dataset.from_dict(train_data).train_test_split(test_size=0.2, seed=42)
    test_dataset = Dataset.from_dict(test_data)
    return train_dataset, test_dataset

def main(batch_size=128, lr= 2e-5, weight_decay=0.01, save_total_limit=1, num_train_epochs=5, model_name="models/pegasus-best"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    train_dataset, test_dataset = create_dataset()


    def tokenize_function(example):
        text_tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=60)
        label_tokens = tokenizer(example['label'], padding='max_length', truncation=True, max_length=60)
        
        text_tokens["labels"] = label_tokens["input_ids"]
        
        return text_tokens

    tokenized_traindataset = train_dataset.map(tokenize_function, batched=True)

    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        report_to='tensorboard',
        seed = 42,
    )


    def compute_metric(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return calculate_metric(decoded_labels, decoded_preds)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_traindataset['train'],
        eval_dataset=tokenized_traindataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )

    trainer.train()
    # # saving model
    trainer.save_model('models/pegasus-finetuned')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for training")
 
    parser.add_argument("--batch_size", help="batch size", required=True)
    parser.add_argument("--lr", help="learning rate", required=True)
    parser.add_argument("--wd", help="weight decay", required=True)
    parser.add_argument("--stl", help="save total limit", required=True)
    parser.add_argument("--model", help="model", required=True)
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd, save_total_limit=args.stl, model_name=args.model)