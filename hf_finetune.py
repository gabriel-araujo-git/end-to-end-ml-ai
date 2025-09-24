# hf_finetune.py
"""
Fine-tuning Transformer (text classification) com Hugging Face Trainer.
Requer: transformers, datasets, evaluate
pip install transformers datasets evaluate sentencepiece
"""

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os

MODEL_NAME = "distilbert-base-uncased"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def main(dataset_name="imdb", output_dir="./hf_out", epochs=2):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(ex):
        return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256)
    tokenized = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized['train'].shuffle(seed=42).select(range(5000)),  # exemplo rápido
        eval_dataset=tokenized['test'].shuffle(seed=42).select(range(2000)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("Fine-tuning concluído. Modelo salvo em", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    main(dataset_name=args.dataset, epochs=args.epochs)
