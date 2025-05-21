from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
import time
import argparse


def generate_data(model_path, tokenizer, cache_dir, output_path, shard_index=0, num_examples=1000, num_shards=1):
    # Define 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    # Load and shard the dataset
    raw_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split='train', cache_dir=cache_dir)
    raw_dataset = raw_dataset.shard(num_shards=num_shards, index=shard_index)
    # raw_dataset = raw_dataset.select(range(min(num_examples, len(raw_dataset))))

    prompts = []
    for example in tqdm(raw_dataset, desc="Creating prompts"):
        # Skip multi-turn conversations
        msgs = example["messages"]
        user_messages = [m for m in msgs if m["role"] == "user"]
        if len(user_messages) > 1:
            continue

        # Remove last assistant message if present
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Skip if prompt is too long
        tokens = tokenizer(prompt, return_tensors="pt")
        if len(tokens.input_ids[0]) > 2048:
            continue

        prompts.append(prompt)

    max_new_tokens = 4096
    new_data = []
    for prompt, orig_example in tqdm(zip(prompts, raw_dataset), total=len(prompts), desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = (inputs["attention_mask"][0] == 1).sum().item()
        gen_tokens = outputs[0][input_len:]
        # Discard if generation likely hit max_new_tokens
        if len(gen_tokens) >= max_new_tokens:
            continue
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        new_messages = orig_example["messages"] + [{"role": "assistant", "content": generated.strip()}]
        new_data.append({"messages": new_messages})

    # Save the new dataset
    new_dataset = Dataset.from_list(new_data)
    new_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--worker_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="synthetic_dataset")
    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)

    generate_data(
        model_path=model_path,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        shard_index=args.worker_idx,
        num_examples=1000,
        num_shards=args.num_shards
    )
