from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import time

def load_data(tokenizer, seed=None, cache_dir="cache", subset_index=0, num_examples=10000, split="train"):

    def process_batch(batch):
        # Just use the "text" field (no EOS)
        texts = [text for text in batch["text"]]
        return {"text": texts}

    # Load the filtered dataset from the hub
    raw_dataset = load_dataset(
        "BarryFutureman/filtered-tinystories-1k-gpt2",
        split=split,
        cache_dir=cache_dir
    )

    # Shuffle for reproducibility
    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 32)
    raw_dataset = raw_dataset.shuffle(seed=seed)

    print("Split", split, raw_dataset[0])

    # Select up to num_examples
    raw_dataset = raw_dataset.select(range(min(num_examples, len(raw_dataset))))

    # Map processing
    processed_dataset = raw_dataset.map(
        process_batch,
        remove_columns=raw_dataset.column_names,
        desc="Processing cosmopedia dataset",
        batched=True,
        batch_size=64,
        num_proc=1
    )

    return processed_dataset

def push_filtered_dataset_to_hub(tokenizer, repo_name, token, cache_dir="cache"):

    def process_batch(batch):
        texts = [text for text in batch["text"]]
        return {"text": texts}

    raw_dataset = load_dataset("fhswf/TinyStoriesV2_cleaned", split="train", cache_dir=cache_dir)

    def token_length_filter(batch):
        return [len(tokens) < 1000 for tokens in tokenizer(batch["text"], add_special_tokens=False)["input_ids"]]

    raw_dataset = raw_dataset.filter(token_length_filter, batched=True, batch_size=512, num_proc=1)
    print(f"Loaded {len(raw_dataset)} examples from the dataset that are < 1000 tokens.")

    val_size = 4000
    train_dataset = raw_dataset.select(range(val_size, len(raw_dataset)))
    val_dataset = raw_dataset.select(range(val_size))

    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

    train_dataset = train_dataset.map(
        process_batch,
        remove_columns=train_dataset.column_names,
        desc="Processing train split",
        batched=True,
        batch_size=64,
        num_proc=1
    )
    val_dataset = val_dataset.map(
        process_batch,
        remove_columns=val_dataset.column_names,
        desc="Processing val split",
        batched=True,
        batch_size=64,
        num_proc=1
    )

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset
    })

    print("Pushing to hub...")
    dataset_dict.push_to_hub(repo_name, token=token)
    print("Done.")