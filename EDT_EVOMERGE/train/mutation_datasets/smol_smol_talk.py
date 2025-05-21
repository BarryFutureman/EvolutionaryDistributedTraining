from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset, Dataset
import time


def load_data(tokenizer, seed=None, cache_dir="cache", subset_index=0, split="train", num_examples=10000):
    EOS_TOKEN = tokenizer.eos_token

    def process_batch(batch):
        # Parse the input field as a list of dicts
        prompts = batch["messages"]
        # Apply chat template
        contexts = [
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            for prompt in prompts
        ]
        contexts = [context + EOS_TOKEN for context in contexts]

        return {
            "text": contexts
        }

    # Load the dataset
    if split == "train":
        raw_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split='train', cache_dir=cache_dir)
    else:
        raw_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split='test', cache_dir=cache_dir)

    # Shuffle the filtered dataset
    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 32)
    raw_dataset = raw_dataset.shuffle(seed=seed)

    raw_dataset = raw_dataset.select(range(min(num_examples * 4, len(raw_dataset))))

    # Tokenize to get lengths for binning
    def get_length(batch):
        contexts = []
        for msgs in batch["messages"]:
            context = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            contexts.append(context)
        tokenized = tokenizer(contexts)
        return {"num_tokens": [len(ids) for ids in tokenized["input_ids"]]}

    tokenized_dataset = raw_dataset.map(get_length,
                                        desc="Calculating token lengths",
                                        batched=True, batch_size=64, num_proc=4)

    # Define bins
    bins = [
        (0, 768),
        (768, 1128),
        (1128, 1444),
        (1200, 2048),
        (0, 2048),
    ]

    # Filter for the subset_index bin
    bin_start, bin_end = bins[subset_index]
    filtered_dataset = tokenized_dataset.filter(
        lambda x: bin_start <= x["num_tokens"] < bin_end,
        desc=f"Filtering for bin {bin_start}-{bin_end}"
    )

    filtered_dataset = filtered_dataset.select(range(min(num_examples, len(filtered_dataset))))

    # Map the processing function in batch
    processed_dataset = filtered_dataset.map(
        process_batch,
        remove_columns=filtered_dataset.column_names,
        desc="Processing dataset",
        batched=True,  # This enables batch processing, required for multiprocessing
        batch_size=64,  # Adjust based on your memory and processing needs
        num_proc=4  # Number of processes to use (adjust based on your CPU cores)
    )

    return processed_dataset


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", cache_dir="../../cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Print total count of examples for each shard
    max_shards = 4
    for shard_index in range(max_shards):
        dataset = load_data(tokenizer, subset_index=shard_index, num_examples=1000, cache_dir="../../cache")
        print(f"Shard {shard_index}: {len(dataset)} examples")
        print(dataset[0])
        quit()
