from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain
import os

# Tokenization function 
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

# Grouping function
def group_texts(examples):
    # Concatenate all texts in the batch
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    
    # Set the block size
    block_size = 512
    
    # We drop the small remainder; we could add padding if we wanted instead of this drop
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

num_proc = 32

# 3. Load the Wikipedia dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(f"Original dataset size: {len(dataset)}")

# 4. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 5. Tokenize the dataset without truncation
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=num_proc, 
    remove_columns=dataset.column_names
)

# 6. Group the tokenized texts
tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=num_proc,
)

# 7. Shuffle the dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34) 
print(f"Tokenized dataset size after grouping: {len(tokenized_datasets)}")

# 8. Save the processed dataset
tokenized_datasets.save_to_disk("tokenized_wikipedia_20231101.hf")
