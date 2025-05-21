from transformers import AutoTokenizer
from datasets import load_dataset

def load_data(tokenizer, seed=69, cache_dir="cache", num_examples=1000, batch_size=16):
    def process_batch(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for context in batch["messages"]:
            # context is already in apply_chat_template-ready format (list of dicts)
            chat_str = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(
                chat_str,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=4096
            )
            input_tokens = tokenized["input_ids"].squeeze().tolist()
            attention_mask = tokenized["attention_mask"].squeeze().tolist()

            # Prepare labels: only supervise assistant responses
            label = [-100] * len(input_tokens)
            current_position = 0
            for turn in context:
                role = turn.get("role", "")
                content = turn.get("content", "")
                content_tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
                for i in range(current_position, len(input_tokens) - len(content_tokens) + 1):
                    if input_tokens[i:i+len(content_tokens)] == content_tokens:
                        if role == "assistant":
                            label[i:i+len(content_tokens)] = content_tokens
                        current_position = i + len(content_tokens)
                        break

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

    raw_dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split='train', cache_dir=cache_dir)
    raw_dataset = raw_dataset.shuffle(seed=seed)
    raw_dataset = raw_dataset.select(range(min(num_examples, len(raw_dataset))))

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
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", cache_dir="cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_data(tokenizer, num_examples=1000, batch_size=8)

    print(len(dataset))
    for example in dataset["input_ids"][:10]:
        print(example)
        print("=" * 50)