import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from utils import ClinicalNotesDataset

# Load the preprocessed data
train_cancer_df = pd.read_csv('data/train_cancer.csv')
val_cancer_df = pd.read_csv('data/val_cancer.csv')
train_diabetes_df = pd.read_csv('data/train_diabetes.csv')
val_diabetes_df = pd.read_csv('data/val_diabetes.csv')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortronS')

# Define constants
max_len = 512
batch_size = 2

# Create datasets
train_cancer_dataset = ClinicalNotesDataset(
    texts=train_cancer_df['text'].tolist(),
    labels=train_cancer_df['has_cancer'].tolist(),
    tokenizer=tokenizer,
    max_len=max_len
)

val_cancer_dataset = ClinicalNotesDataset(
    texts=val_cancer_df['text'].tolist(),
    labels=val_cancer_df['has_cancer'].tolist(),
    tokenizer=tokenizer,
    max_len=max_len
)

train_diabetes_dataset = ClinicalNotesDataset(
    texts=train_diabetes_df['text'].tolist(),
    labels=train_diabetes_df['has_diabetes'].tolist(),
    tokenizer=tokenizer,
    max_len=max_len
)

val_diabetes_dataset = ClinicalNotesDataset(
    texts=val_diabetes_df['text'].tolist(),
    labels=val_diabetes_df['has_diabetes'].tolist(),
    tokenizer=tokenizer,
    max_len=max_len
)

# Define training arguments with adjusted parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increase the number of epochs due to small dataset size
    per_device_train_batch_size=batch_size,  # Small batch size due to limited data
    per_device_eval_batch_size=batch_size,
    warmup_steps=50,  # Reduce warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,  # Increase logging steps to avoid too frequent logging
    evaluation_strategy="steps",
    eval_steps=10,  # Adjust eval steps to fit the small dataset
    save_steps=10,  # Adjust save steps to fit the small dataset
    gradient_accumulation_steps=4,  # Simulate larger batch size
    save_total_limit=2  # Limit the number of saved checkpoints
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define and train the cancer model
model_cancer = AutoModelForSequenceClassification.from_pretrained('UFNLP/gatortronS', num_labels=2)
trainer_cancer = Trainer(
    model=model_cancer,
    args=training_args,
    train_dataset=train_cancer_dataset,
    eval_dataset=val_cancer_dataset,
    data_collator=data_collator
)
trainer_cancer.train()

# Save the cancer model
model_cancer.save_pretrained('./results/cancer_model')

# Define and train the diabetes model
model_diabetes = AutoModelForSequenceClassification.from_pretrained('UFNLP/gatortronS', num_labels=2)
trainer_diabetes = Trainer(
    model=model_diabetes,
    args=training_args,
    train_dataset=train_diabetes_dataset,
    eval_dataset=val_diabetes_dataset,
    data_collator=data_collator
)
trainer_diabetes.train()

# Save the diabetes model
model_diabetes.save_pretrained('./results/diabetes_model')
