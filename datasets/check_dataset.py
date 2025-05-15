from datasets import load_dataset
from transformers import AutoTokenizer
from statistics import mean
import numpy as np


def safe_mean(values):
    return f"{mean(values):.1f}" if values else "n/a"


def safe_max(values):
    return max(values) if values else "n/a"


def format_prompt(messages: list[dict]) -> str:
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted


def format_messages(messages):
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def print_percentiles(name, values):
    if not values:
        print(f"{name}: нет данных.")
        return
    percentiles = [50, 75, 90, 95, 99]
    stats = np.percentile(values, percentiles)
    print(f"\n{name}:")
    for p, val in zip(percentiles, stats):
        print(f"  {p:>2} перцентиль: {int(val)}")


def safe_mean(x):
    return f"{mean(x):.1f}" if x else "n/a"


def safe_max(x):
    return max(x) if x else "n/a"


dataset_name = "IlyaGusev/saiga_preferences"
split = "train"
model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
max_rows = 100000

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
ds = load_dataset(dataset_name, split=split)

prompt_lengths, chosen_lengths, rejected_lengths, total_lengths = [], [], [], []

for row in ds.select(range(min(max_rows, len(ds)))):
    for k in ["chosen", "rejected", "prompt"]:
        if isinstance(row[k], str):
            row[k] = [
                {"role": "assistant" if k != "prompt" else "user", "content": row[k]}
            ]

    if not all(
        isinstance(row[k], list) and row[k] for k in ["prompt", "chosen", "rejected"]
    ):
        continue

    try:
        prompt_text = format_messages(row["prompt"])
        chosen_text = format_messages(row["chosen"])
        rejected_text = format_messages(row["rejected"])

        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        chosen_len = len(tokenizer.encode(chosen_text, add_special_tokens=False))
        rejected_len = len(tokenizer.encode(rejected_text, add_special_tokens=False))

        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        total_lengths.append(prompt_len + max(chosen_len, rejected_len))
    except Exception:
        continue


print("Prompt:")
print(f"Средняя длина: {safe_mean(prompt_lengths)}")
print(f"Макс. длина: {safe_max(prompt_lengths)}")

print("Completion:")
print(f"Средняя длина chosen: {safe_mean(chosen_lengths)}")
print(f"Средняя длина rejected: {safe_mean(rejected_lengths)}")
print(f"Макс.: {safe_max(chosen_lengths + rejected_lengths)}")

print("Общая длина:")
print(f"Средняя: {safe_mean(total_lengths)}")
print(f"Макс.: {safe_max(total_lengths)}")


print_percentiles("Prompt", prompt_lengths)
print_percentiles("Chosen completion", chosen_lengths)
print_percentiles("Rejected completion", rejected_lengths)
print_percentiles("Prompt + max(completion)", total_lengths)
