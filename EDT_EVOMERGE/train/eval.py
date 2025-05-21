import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from tqdm import tqdm
import random # Import random for sampling
from collections import defaultdict # Import defaultdict
import json


CACHE_DIR = "cache"
SEED = 0

# --- Configuration ---
MMLU_DATASET_NAME = "TIGER-Lab/MMLU-Pro"
AIME_DATASET_NAME = "Maxwell-Jia/AIME_2024"
GSM8K_DATASET_NAME = "openai/gsm8k"
HELLASWAG_DATASET_NAME = "PleIAs/GoldenSwag"
BBH_DATASET_NAME = "lukaemon/bbh"
DATASET_SPLIT = "test" # Or 'validation' if preferred ('validation' for HellaSwag)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "You are a helpful assistant. Answer the following question. Place the final answer inside \\boxed{{}}. For multiple-choice questions, place only the option letter inside \\boxed{{}}."
MMLU_SAMPLES_PER_CATEGORY = 0
MMLU_TARGET_CATEGORIES = ['business', 'law', 'psychology', 'biology', 'chemistry', 'history', 'other', 'health', 'economics', 'math', 'physics', 'computer science', 'philosophy', 'engineering']
AIME_SAMPLES = 0
GSM8K_SAMPLES = 8
HELLASWAG_SAMPLES = 8
BBH_SAMPLES_PER_CATEGORY = 0
BBH_TARGET_CATEGORIES = [
    'boolean_expressions',
    'causal_judgement',
    'date_understanding',
    'disambiguation_qa',
    # 'dyck_languages',
    'formal_fallacies',
    # 'geometric_shapes',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'logical_deduction_three_objects',
    'movie_recommendation',
    'multistep_arithmetic_two',
    'navigate',
    'object_counting',
    'penguins_in_a_table',
    'reasoning_about_colored_objects',
    'ruin_names',
    'salient_translation_error_detection',
    'snarks',
    'sports_understanding',
    'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects',
    'web_of_lies',
    'word_sorting'
]


def extract_answer(text: str) -> str:
    """
    Extracts the content from the last \boxed{}
    """
    matches = re.findall(r"\\boxed{(.*?)}", text)
    if matches:
        answer = matches[-1].strip()
        # Try to extract only numbers and dot, then convert to float/int
        numeric_part = ''.join(c for c in answer if c.isdigit() or c == '.').strip('.')
        if numeric_part:
            try:
                num = float(numeric_part)
                if num.is_integer():
                    return str(int(num))
                else:
                    return str(num)
            except ValueError:
                pass
        # Fallback to current method
        # If answer is wrapped in \text{...}, extract the inner text
        text_match = re.match(r"\\text{(.*)}", answer)
        if text_match:
            answer = text_match.group(1).strip()
        # Remove dollar signs
        answer = answer.replace("$", "").strip()
        # Strip parentheses
        if answer.startswith("(") and answer.endswith(")"):
            answer = answer[1:-1].strip()
        # Try to convert to float, then int if possible
        try:
            num = float(answer)
            if num.is_integer():
                return str(int(num))
            else:
                return str(num)
        except ValueError:
            pass
        return answer
    return ""  # Return empty string if no valid answer is found


# --- Function to Load and Prepare MMLU Data ---
def load_and_prepare_mmlu(dataset_name, split, categories, samples_per_category, tokenizer, system_prompt):
    print(f"Loading MMLU dataset: {dataset_name} ({split} split)")
    original_dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
    print(f"Original MMLU dataset loaded with {len(original_dataset)} examples.")

    print(f"Sampling {samples_per_category} examples per category from: {categories}")
    sampled_datasets = []
    for category in categories:
        category_dataset = original_dataset.filter(lambda example: example['category'] == category)
        num_samples = len(category_dataset)

        if num_samples == 0:
            print(f"Warning: No MMLU examples found for category '{category}'. Skipping.")
            continue

        sample_size = min(samples_per_category, num_samples)
        print(f"Sampling {sample_size} MMLU examples for category '{category}' (found {num_samples}).")
        sampled_datasets.append(category_dataset.shuffle(seed=SEED).select(range(sample_size)))

    if not sampled_datasets:
        print("Warning: No MMLU examples found for any of the target categories.")
        return None # Return None if no MMLU data could be sampled

    mmlu_dataset = concatenate_datasets(sampled_datasets)
    print(f"MMLU dataset contains {len(mmlu_dataset)} examples after sampling.")
    del original_dataset, sampled_datasets # Free up memory

    def format_mmlu_prompt(example):
        """Formats the question and options for the MMLU task."""
        question = example['question']
        options = example['options']
        option_labels = [chr(ord('A') + i) for i in range(len(options))]
        formatted_options = "\n".join([f"{label}. {option}" for label, option in zip(option_labels, options)])
        user_prompt = f"{question}\nOptions:\n{formatted_options}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) # Debug print
        return {
            "context": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": chr(ord('A') + example['answer_index']),
            "dataset_source": "MMLU" # Add dataset source
        }

    # Keep original columns needed for mapping, then remove
    original_columns = mmlu_dataset.column_names
    mmlu_dataset = mmlu_dataset.map(format_mmlu_prompt, remove_columns=original_columns)
    return mmlu_dataset

# --- Function to Load and Prepare AIME Data ---
def load_and_prepare_aime(dataset_name, split, tokenizer, system_prompt, num_samples):
    print(f"Loading AIME dataset: {dataset_name} ({split} split)")
    try:
        aime_dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
        print(f"Original AIME dataset loaded with {len(aime_dataset)} examples.")
    except Exception as e:
        print(f"Error loading AIME dataset: {e}")
        return None

    # Sample AIME data if needed
    if num_samples is not None and num_samples < len(aime_dataset):
        print(f"Sampling {num_samples} examples from AIME dataset.")
        aime_dataset = aime_dataset.shuffle(seed=SEED).select(range(num_samples))
    else:
        print(f"Using all {len(aime_dataset)} examples from AIME dataset.")

    if num_samples is not None and num_samples == 0:
        print("Skipping AIME dataset as sample count is 0.")
        return None

    def format_aime_prompt(example):
        """Formats the problem for the AIME task."""
        user_prompt = f"Solve the following math problem:\n{example['Problem']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) # Debug print
        return {
            "context": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": str(example['Answer']), # Ensure answer is string
            "dataset_source": "AIME" # Add dataset source
        }

    # Keep original columns needed for mapping, then remove
    original_columns = aime_dataset.column_names
    aime_dataset = aime_dataset.map(format_aime_prompt, remove_columns=original_columns)
    return aime_dataset

# --- Function to Load and Prepare GSM8K Data ---
def load_and_prepare_gsm8k(dataset_name, split, tokenizer, system_prompt, num_samples):
    print(f"Loading GSM8K dataset: {dataset_name} (main, {split} split)")
    try:
        # GSM8K uses 'main' config and has 'train' and 'test' splits
        gsm8k_dataset = load_dataset(dataset_name, 'main', split=split, cache_dir=CACHE_DIR)
        print(f"Original GSM8K dataset loaded with {len(gsm8k_dataset)} examples.")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        return None

    # Sample GSM8K data if needed
    if num_samples is not None and num_samples < len(gsm8k_dataset):
        print(f"Sampling {num_samples} examples from GSM8K dataset.")
        gsm8k_dataset = gsm8k_dataset.shuffle(seed=SEED).select(range(num_samples))
    else:
        print(f"Using all {len(gsm8k_dataset)} examples from GSM8K dataset.")

    if num_samples is not None and num_samples == 0:
        print("Skipping GSM8K dataset as sample count is 0.")
        return None

    def extract_gsm8k_answer(text):
        # The answer is the number after the last '####'
        parts = text.split('####')
        if len(parts) > 1:
            return parts[-1].strip()
        return "" # Should not happen for GSM8K format

    def format_gsm8k_prompt(example):
        """Formats the question for the GSM8K task."""
        user_prompt = f"Solve the following math word problem:\n{example['question']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return {
            "context": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": extract_gsm8k_answer(example['answer']), # Extract final numerical answer
            "dataset_source": "GSM8K" # Add dataset source
        }

    # Keep original columns needed for mapping, then remove
    original_columns = gsm8k_dataset.column_names
    gsm8k_dataset = gsm8k_dataset.map(format_gsm8k_prompt, remove_columns=original_columns)
    return gsm8k_dataset

# --- Function to Load and Prepare HellaSwag Data ---
def load_and_prepare_hellaswag(dataset_name, split, tokenizer, system_prompt, num_samples):
    print(f"Loading HellaSwag dataset: {dataset_name} ({split} split)")
    try:
        # HellaSwag uses 'validation' split for evaluation
        hellaswag_dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
        print(f"Original HellaSwag dataset loaded with {len(hellaswag_dataset)} examples.")
    except Exception as e:
        print(f"Error loading HellaSwag dataset: {e}")
        return None

    # Sample HellaSwag data if needed
    if num_samples is not None and num_samples < len(hellaswag_dataset):
        print(f"Sampling {num_samples} examples from HellaSwag dataset.")
        hellaswag_dataset = hellaswag_dataset.shuffle(seed=SEED).select(range(num_samples))
    else:
        print(f"Using all {len(hellaswag_dataset)} examples from HellaSwag dataset.")

    if num_samples is not None and num_samples == 0:
        print("Skipping HellaSwag dataset as sample count is 0.")
        return None

    def format_hellaswag_prompt(example):
        """Formats the context and endings for the HellaSwag task."""
        context = example['ctx']
        endings = example['endings']
        option_labels = [chr(ord('A') + i) for i in range(len(endings))]
        formatted_options = "\n".join([f"{label}. {ending}" for label, ending in zip(option_labels, endings)])
        # Modify the user prompt slightly for clarity
        user_prompt = f"Complete the following sentence by choosing the most logical ending:\n\n{context}\n\nOptions:\n{formatted_options}\n\nWhich ending (A, B, C, or D) is the most logical continuation?"
        messages = [
            {"role": "system", "content": system_prompt}, # Use the same system prompt asking for \boxed{}
            {"role": "user", "content": user_prompt}
        ]
        correct_label_index = int(example['label']) # Label is the index of the correct ending
        correct_label = chr(ord('A') + correct_label_index)
        return {
            "context": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": correct_label, # The answer is the letter (A, B, C, D)
            "dataset_source": "HellaSwag" # Add dataset source
        }

    # Keep original columns needed for mapping, then remove
    original_columns = hellaswag_dataset.column_names
    hellaswag_dataset = hellaswag_dataset.map(format_hellaswag_prompt, remove_columns=original_columns)
    return hellaswag_dataset

# --- Function to Load and Prepare BBH Data ---
def load_and_prepare_bbh(dataset_name, split, categories, samples_per_category, tokenizer, system_prompt):
    print(f"Loading BBH dataset: {dataset_name} ({split} split) for categories: {categories}")

    # Ensure categories are provided and sampling is requested
    if not categories or samples_per_category is None or samples_per_category <= 0:
        print("Skipping BBH dataset: No categories specified or samples_per_category is not positive.")
        return None

    sampled_datasets = []
    print(f"Sampling {samples_per_category} examples per category from BBH categories.")
    for category in categories:
        try:
            # Load each category (config) individually
            category_dataset = load_dataset(dataset_name, name=category, split=split, cache_dir=CACHE_DIR)
            num_samples = len(category_dataset)
            print(f"Loaded BBH category '{category}' with {num_samples} examples.")

            if num_samples == 0:
                print(f"Warning: No BBH examples found for category '{category}'. Skipping.")
                continue

            sample_size = min(samples_per_category, num_samples)
            print(f"Sampling {sample_size} BBH examples for category '{category}'.")
            # Add category info to the example for potential later use if needed
            # category_dataset = category_dataset.map(lambda example: {'task_name': category}) # Optional
            sampled_datasets.append(category_dataset.shuffle(seed=SEED).select(range(sample_size)))

        except Exception as e:
            print(f"Error loading or sampling BBH category '{category}': {e}")
            continue # Skip this category if loading fails

    if not sampled_datasets:
        print("Warning: No BBH examples could be loaded or sampled for any of the target categories.")
        return None # Return None if no BBH data could be sampled

    bbh_dataset = concatenate_datasets(sampled_datasets)
    print(f"BBH dataset contains {len(bbh_dataset)} examples after sampling.")
    del sampled_datasets # Free up memory

    def format_bbh_prompt(example):
        """Formats the input for the BBH task."""
        # The 'input' field usually contains the question and context/options
        user_prompt = example['input']
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # The 'target' field contains the expected answer string
        target_answer = str(example['target']).strip()
        # Check if the target answer is enclosed in parentheses and strip them
        if target_answer.startswith("(") and target_answer.endswith(")"):
            target_answer = target_answer[1:-1].strip()
        return {
            "context": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": target_answer, # Use the potentially stripped answer
            "dataset_source": "BBH" # Add dataset source
        }

    # Keep original columns needed for mapping, then remove
    original_columns = bbh_dataset.column_names
    bbh_dataset = bbh_dataset.map(format_bbh_prompt, remove_columns=original_columns)
    return bbh_dataset


def eval_main(model_paths, seed=42, batch_size=32, cache_dir="cache"):
    random.seed(seed)
    torch.manual_seed(seed)

    global CACHE_DIR, SEED
    CACHE_DIR = cache_dir
    SEED = seed

    print(model_paths)

    # --- Load Tokenizer for Data Preparation ---
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load and Combine Datasets (only once) ---
    mmlu_data = load_and_prepare_mmlu(
        MMLU_DATASET_NAME, DATASET_SPLIT, MMLU_TARGET_CATEGORIES, MMLU_SAMPLES_PER_CATEGORY, tokenizer, SYSTEM_PROMPT
    )
    aime_data = load_and_prepare_aime(
        AIME_DATASET_NAME, "train", tokenizer, SYSTEM_PROMPT, AIME_SAMPLES
    )
    gsm8k_data = load_and_prepare_gsm8k(
        GSM8K_DATASET_NAME, DATASET_SPLIT, tokenizer, SYSTEM_PROMPT, GSM8K_SAMPLES
    )
    hellaswag_data = load_and_prepare_hellaswag(
        HELLASWAG_DATASET_NAME, "validation", tokenizer, SYSTEM_PROMPT, HELLASWAG_SAMPLES
    )
    bbh_data = load_and_prepare_bbh(
        BBH_DATASET_NAME, "test", BBH_TARGET_CATEGORIES, BBH_SAMPLES_PER_CATEGORY, tokenizer, SYSTEM_PROMPT
    )

    datasets_to_combine = []
    dataset_sources_loaded = []

    if mmlu_data:
        datasets_to_combine.append(mmlu_data)
        dataset_sources_loaded.append("MMLU")
    if aime_data:
        datasets_to_combine.append(aime_data)
        dataset_sources_loaded.append("AIME")
    if gsm8k_data:
        datasets_to_combine.append(gsm8k_data)
        dataset_sources_loaded.append("GSM8K")
    if hellaswag_data:
        datasets_to_combine.append(hellaswag_data)
        dataset_sources_loaded.append("HellaSwag")
    if bbh_data:
        datasets_to_combine.append(bbh_data)
        dataset_sources_loaded.append("BBH")

    if not datasets_to_combine:
        raise ValueError("No data loaded from any dataset. Exiting.")

    dataset = concatenate_datasets(datasets_to_combine)
    print(f"Final combined dataset contains {len(dataset)} examples from sources: {dataset_sources_loaded}")
    dataset = dataset.shuffle(seed=seed)

    # --- Evaluate Each Model ---
    for MODEL_NAME in model_paths:
        print(f"\n\n=== Evaluating model: {MODEL_NAME} ===")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"Loading model: {MODEL_NAME} with cache_dir: {cache_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )
        model.eval()
        print(f"Model and tokenizer loaded on device: {model.device}")

        correct_predictions = 0
        total_predictions = 0
        correct_predictions_per_source = defaultdict(int)
        total_predictions_per_source = defaultdict(int)

        print("Starting evaluation on combined dataset...")
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch_data = dataset[batch_indices]

            prompts = batch_data['context']
            ground_truth_labels = batch_data['answer']
            dataset_sources = batch_data['dataset_source']

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_k=20,
                    top_p=0.95,
                )

            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for j, decoded_text in enumerate(decoded_outputs):
                predicted_label = extract_answer(decoded_text)
                ground_truth = ground_truth_labels[j]
                source = dataset_sources[j]

                total_predictions += 1
                total_predictions_per_source[source] += 1

                if predicted_label and predicted_label == ground_truth:
                    correct_predictions += 1
                    correct_predictions_per_source[source] += 1
                
                print("Model output:", decoded_text)
                print(f"Example {i+j}: Predicted: {predicted_label}, Ground Truth: {ground_truth}")

        print("\n--- Evaluation Complete ---")
        print(f"Model: {MODEL_NAME}")
        print(f"MMLU: {MMLU_DATASET_NAME}, {MMLU_SAMPLES_PER_CATEGORY} samples/cat from {len(MMLU_TARGET_CATEGORIES)} cats")
        print(f"AIME: {AIME_DATASET_NAME}, {AIME_SAMPLES} samples")
        print(f"GSM8K: {GSM8K_DATASET_NAME}, {GSM8K_SAMPLES} samples")
        print(f"HellaSwag: {HELLASWAG_DATASET_NAME}, {HELLASWAG_SAMPLES} samples")
        print(f"BBH: {BBH_DATASET_NAME}, {BBH_SAMPLES_PER_CATEGORY} samples/cat from {len(BBH_TARGET_CATEGORIES)} cats")

        print("\n--- Accuracy Results ---")
        for source in dataset_sources_loaded:
            source_correct = correct_predictions_per_source[source]
            source_total = total_predictions_per_source[source]
            source_accuracy = source_correct / source_total if source_total > 0 else 0
            print(f"Accuracy ({source}): {source_accuracy:.4f} ({source_correct}/{source_total})")

        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Total Examples Evaluated: {total_predictions}")

        # --- Write fitness to genome.json ---
        genome_path = f"{MODEL_NAME}/genome.json"
        try:
            with open(genome_path, "r") as f:
                genome = json.load(f)
        except Exception:
            genome = {}

        genome["fitness"] = max(overall_accuracy, 0.1)

        with open(genome_path, "w") as f:
            json.dump(genome, f, indent=4)
            
        del model, tokenizer
        torch.cuda.empty_cache()
        
    print("\n\n=== Evaluation Complete for All Models ===")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model(s) and update genome fitness.")
    parser.add_argument("--model_path", type=str, required=True, nargs="+", help="Path(s) to model directory(s)")
    parser.add_argument("--genome_path", type=str, required=False, help="Path to genome.json (not used directly)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="cache")
    args = parser.parse_args()

    # Accepts one or more model paths
    if isinstance(args.model_path, str):
        model_paths = [args.model_path]
    else:
        model_paths = args.model_path

    eval_main(
        model_paths=model_paths,
        seed=args.seed,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir
    )
