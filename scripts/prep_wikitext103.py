import re
from datasets import load_dataset
import random
from transformers import AutoTokenizer

def clean_wikitext(example):
    text = example['text']
    # Remove headers
    text = re.sub(r'= .+ =', '', text)
    # Remove lists
    text = re.sub(r'^\s*[\*\-]\s*.+$', '', text, flags=re.MULTILINE)
    # Remove tables
    text = re.sub(r'{\|[\s\S]*?\|}', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text).strip()
    # Replace @ @
    text = text.replace('@.@', '.').replace('@,@', ',').replace('@-@', '-')
    return {'text': text}

def remove_empty_sequences(example):
    return len(example['text'].strip()) > 0

def display_samples(dataset, num_samples=5):
    samples = random.sample(range(len(dataset["train"])), num_samples)
    dataset_split = dataset["train"] 
    for i, idx in enumerate(samples, 1):
        print(f"\nSample {i}:")
        text = dataset_split[idx]['text']
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 80)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Clean the dataset
cleaned_dataset = dataset.map(clean_wikitext)

# Remove empty sequences
filtered_dataset  = cleaned_dataset.filter(remove_empty_sequences) #Keep sequence if > 0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=filtered_dataset["train"].column_names)

display_samples(filtered_dataset)

tokenized_datasets.save_to_disk("tokenized_preprocessed_wikitext103.hf")


