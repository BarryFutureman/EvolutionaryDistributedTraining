import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_constant_schedule, get_linear_schedule_with_warmup
import os

def prepare_models(model_name, cache_dir="cache"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    # Create optimizer and constant scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    
    # Try to load optimizer and scheduler state if they exist
    optimizer_path = os.path.join(model_name, "optimizer.pt")
    scheduler_path = os.path.join(model_name, "scheduler.pt")
    if os.path.exists(optimizer_path):
        print(f"Loading optimizer state from {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        
    # if os.path.exists(scheduler_path):
    #     print(f"Loading scheduler state from {scheduler_path}")
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=4000
    #     )
    #     print("torch.load(scheduler_path)", torch.load(scheduler_path, map_location="cpu"))
    #     scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
    # else:
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=4000
    #     )
    scheduler = get_constant_schedule(optimizer)


    return model, tokenizer, optimizer, scheduler

def run_training(model, base_model_path, tokenizer, processed_dataset, output_dir, batch_multiplier=1, optimizer=None, scheduler=None):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import wandb
    import os

    if batch_multiplier > 2:
        gradient_accumulation_steps = 4
        batch_multiplier = batch_multiplier // gradient_accumulation_steps
    else:
        gradient_accumulation_steps = 1

    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        dataloader_drop_last=True,
        save_strategy="no",
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=batch_multiplier,
        learning_rate=4e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=0.1,
        push_to_hub=False,
        include_tokens_per_second=True,
        # report_to="wandb",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        args=training_args,
        optimizers=(optimizer, scheduler) if optimizer is not None and scheduler is not None else (None, None),
    )

    trainer_stats = trainer.train()
    trainer.save_model(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")

    # Save optimizer state
    if optimizer is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(optimizer.state_dict(), f"{output_dir}/optimizer.pt")

    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"{output_dir}/scheduler.pt")
        print(f"saving scheduler state dict at {output_dir}/scheduler.pt")


def evaluate_model(model, tokenizer, val_data, eval_batch_size=16):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import math
    
    eval_args = TrainingArguments(
        output_dir="cache",
        per_device_eval_batch_size=eval_batch_size,
        bf16=True,
        report_to="none",
    )

    # Initialize trainer for evaluation
    eval_trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=val_data,
        eval_dataset=val_data,
    )

    # Run evaluation
    metrics = eval_trainer.evaluate()
    eval_loss = metrics['eval_loss']
    mean_perplexity = math.exp(eval_loss)
    return eval_loss


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
    OUTPUT_DIR = "smollm2-135M-post"
    model, tokenizer = prepare_models(model_name, cache_dir="cache")

    # Load validation data
    from mutation_datasets import smol_smol_talk

    val_dataset = smol_smol_talk.load_data(
        tokenizer=tokenizer,
        bucket_index=1,  # validation shard
        max_buckets=92
    )

    # Evaluate before training
    print("Evaluating pre-training model...")
    pre_train_loss = evaluate_model(model, tokenizer, val_dataset)
    print(f"Pre-training eval loss: {pre_train_loss}")

    smol_smol_talk_dataset = smol_smol_talk.load_data(tokenizer=tokenizer, bucket_index=0, max_buckets=92)
    from datasets import concatenate_datasets

    # Concatenate the datasets
    combined_dataset = concatenate_datasets([smol_smol_talk_dataset])
    processed_dataset = combined_dataset.shuffle(seed=69)
    run_training(model, model_name, tokenizer, processed_dataset, OUTPUT_DIR)

    # Evaluate after training
    print("Evaluating post-training model...")
    post_train_loss = evaluate_model(model, tokenizer, val_dataset)
    print(f"Post-training eval loss: {post_train_loss}")
    print(f"Loss improvement: {pre_train_loss - post_train_loss}")