from tqdm import tqdm
from datasets import load_dataset, Dataset

def load_data(tokenizer, seed=69, cache_dir="cache", num_examples=1000, batch_size=16):
    EOS_TOKEN = tokenizer.eos_token

    def process_batch(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for prompt, output_label in zip(batch["input"], batch["output"]):
            # Apply chat template
            context = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = context + output_label

            # Tokenize (pad to longest in batch)
            tokenized_input = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=4096
            )
            input_tokens = tokenized_input["input_ids"].squeeze().tolist()
            attention_mask = tokenized_input["attention_mask"].squeeze().tolist()

            # Create the labels: Set the context tokens to -100 and use the output label tokens
            label = input_tokens[:]  # Copy of input_ids to create labels

            # Set context tokens (input part) to -100 in the labels
            context_tokens = tokenizer(context, add_special_tokens=False)["input_ids"]
            for i in range(len(input_tokens) - len(context_tokens) + 1):
                if input_tokens[i:i + len(context_tokens)] == context_tokens:
                    label[i:i + len(context_tokens)] = [-100] * len(context_tokens)
                    break  # Only mask the first occurrence of the context
            # Set EOS tokens to -100 in the labels
            for i, token in enumerate(label):
                if token == tokenizer.eos_token_id:
                    label[i] = -100

            input_ids_list.append(input_tokens)
            attention_mask_list.append(attention_mask)
            labels_list.append(label)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    # Load the dataset (no streaming)
    raw_dataset = load_dataset('nvidia/Llama-Nemotron-Post-Training-Dataset', "SFT", split='science', cache_dir=cache_dir)
    raw_dataset = raw_dataset.shuffle(seed=seed)
    raw_dataset = raw_dataset.select(range(min(num_examples, len(raw_dataset))))

    # Map the processing function in batch
    processed_dataset = raw_dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=raw_dataset.column_names,
        desc="Processing dataset",
        num_proc=4
    )

    return processed_dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", cache_dir="cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load dataset
    dataset = load_data(tokenizer, num_examples=1000, batch_size=8)
    
    print(len(dataset))
    # Print all examples
    for example in dataset["input_ids"][:10]:
        print(example)
        print("=" * 50)