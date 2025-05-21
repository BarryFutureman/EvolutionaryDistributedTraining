import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training



def prepare_models(model_name, cache_dir="cache"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2",
    )

    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_alpha=16,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        #                 "gate_proj", "up_proj", "down_proj"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        lora_dropout=0,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def run_training(model, base_model_path, tokenizer, processed_dataset, output_dir, cache_dir="cache", batch_size=16):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import wandb
    
    gradient_accumulation_steps = 4
    batch_size = batch_size // gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir=f"{output_dir}_adapter",
        label_names=["labels"],
        dataloader_drop_last=True,
        # evaluation_strategy="steps", # "steps",
        save_strategy="no",  # "steps",
        num_train_epochs=4,
        # eval_steps=1024,
        # save_steps=128,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=8,
        learning_rate=4e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        push_to_hub=False,
        include_tokens_per_second=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        args=training_args,
    )

    trainer_stats = trainer.train()
    trainer.save_model(f"{output_dir}_adapter")

    # Delete models and free up memory
    del model
    torch.cuda.empty_cache()

    def merge_lora(base_model_path, tokenizer, lora_adapter_path):
        # Load the base model in bf16
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            local_files_only=False,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )

        # Apply the LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)
        model = peft_model.merge_and_unload()

        # Save the merged model
        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")

        torch.cuda.empty_cache()
        return

    # Merge and save the model
    merge_lora(base_model_path, tokenizer, f"{output_dir}_adapter")


def evaluate_model(model, tokenizer, val_data, eval_batch_size=16):
    from trl import SFTTrainer
    from transformers import TrainingArguments
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
    return metrics['eval_loss']


if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    OUTPUT_DIR = "q3-7b-post"
    model, tokenizer = prepare_models(model_name, cache_dir="../cache")

    # Load validation data
    from mutation_datasets import opus_instruct


    opus_instruct_dataset = opus_instruct.load_data(tokenizer=tokenizer, num_examples=100, batch_size=8)
    from datasets import concatenate_datasets

    # Concatenate the datasets
    combined_dataset = concatenate_datasets([opus_instruct_dataset])
    processed_dataset = combined_dataset.shuffle(seed=69)
    run_training(model, model_name, tokenizer, processed_dataset, OUTPUT_DIR, batch_size=8, cache_dir="../cache")