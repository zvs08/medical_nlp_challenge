import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

# Load the CSV file
df = pd.read_csv('data/data-train.csv')

# Filter rows where 'has_cancer' and 'has_diabetes' are not null and 'test_set' is 0
filtered_df = df[df['has_cancer'].notnull() & df['has_diabetes'].notnull() & (df['test_set'] == 0)]

# Split the data into training and validation sets proportionally to the target variables
train_cancer_df, val_cancer_df = train_test_split(
    filtered_df,
    test_size=0.2,
    stratify=filtered_df['has_cancer'],
    random_state=42
)

train_diabetes_df, val_diabetes_df = train_test_split(
    filtered_df,
    test_size=0.2,
    stratify=filtered_df['has_diabetes'],
    random_state=42
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortronS')

# Save the dataframes to CSV for further use
train_cancer_df.to_csv('data/train_cancer.csv', index=False)
val_cancer_df.to_csv('data/val_cancer.csv', index=False)
train_diabetes_df.to_csv('data/train_diabetes.csv', index=False)
val_diabetes_df.to_csv('data/val_diabetes.csv', index=False)
