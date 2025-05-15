import os
import json
import argparse
import random
import codecs
from itertools import chain

import torch
from tqdm import tqdm
import mmh3
import time


from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset as HFDataset, load_from_disk

from trl import CPOTrainer, CPOConfig
from utils import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("cpo_alpha", type=float)
parser.add_argument("base_model_name", type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("sample_fraction", type=float)
args = parser.parse_args()


CONFIG_FILE = "simpo_sp/simpo_config.json"
PREF_CONFIG = "simpo_sp/pref_d38.json"
CUSTOM_TEMPLATE_PATH = "configs/qwen_2.5_instruct_no_system.json"

TRAIN_PATH = "datasets/saiga_pref_selfplay/train.jsonl"
VAL_PATH = "datasets/saiga_pref_selfplay/val.jsonl"


def fix_text(text):
    return text.replace("\xa0", " ").replace("\\xa0", " ").strip()


def is_main_process():
    return (
        int(os.environ.get("RANK", "0")) == 0
        or int(os.environ.get("LOCAL_RANK", "0")) == 0
    )


def wait_for_flag(path, timeout=600):
    for _ in range(timeout):
        if os.path.exists(path):
            return
        time.sleep(1)
    raise TimeoutError(f"Timeout while waiting for flag {path}")


def compose_pref_dataset(config_path: str, train_path: str, val_path: str):
    with open(config_path) as r:
        config = json.load(r)

    sample_rate = args.sample_fraction
    records = []

    # dataset_name = config.get("dataset_name", "IlyaGusev/saiga_preferences")
    # revision = config["dataset_revision"]

    # if isinstance(dataset_name, str):
    #     dataset = load_from_disk("datasets/saiga_preferences")["train"]
    # else:
    #     dataset = chain(
    #         *[
    #             load_dataset(name, split="train", revision=r)
    #             for name, r in zip(dataset_name, revision)
    #         ]
    #     )

    dataset = load_dataset("andrrrei/saiga_pref_selfplay_1_best_sft", split="train")

    field_mapping = config.get("field_mapping", dict())

    for row in dataset:
        if random.random() > sample_rate:
            continue

        if field_mapping:
            for k, v in list(row.items()):
                if k in field_mapping:
                    row[field_mapping[k]] = row.pop(k)

        if config.get("sources") and row["source"] not in config["sources"]:
            continue
        if (
            config.get("chosen_models")
            and row["chosen_model"] not in config["chosen_models"]
        ):
            continue
        if config.get("exclude_regex", False) and row.get("is_bad_by_regex", False):
            continue

        for k in ["chosen", "rejected", "prompt"]:
            if isinstance(row[k], str):
                row[k] = [
                    {
                        "role": "assistant" if k != "prompt" else "user",
                        "content": row[k],
                    }
                ]

        if len(str(row["chosen"])) > len(str(row["rejected"])) * config.get(
            "max_length_ratio", 2.1
        ):
            s = str(row["prompt"]) + str(row["chosen"])
            h = mmh3.hash(s, 1337, signed=False)
            if h % 100 > config.get("max_length_ratio_prob", 0.0) * 100.0:
                continue

        if config.get("sonnet_approved_only", False) and not row.get(
            "sonnet_approved", False
        ):
            continue

        if (
            not row["chosen"][0]["content"].strip()
            or not row["rejected"][0]["content"].strip()
        ):
            continue

        row["chosen"][0]["content"] = fix_text(row["chosen"][0]["content"])
        row["rejected"][0]["content"] = fix_text(row["rejected"][0]["content"])

        for m in row["prompt"]:
            if m["role"] == "bot":
                m["role"] = "assistant"

        records.append(row)

    random.shuffle(records)
    train_records = [
        r for r in records if mmh3.hash(str(r["prompt"]), signed=False) % 100 < 97
    ]
    val_records = [
        r for r in records if mmh3.hash(str(r["prompt"]), signed=False) % 100 >= 97
    ]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w") as w:
        for r in train_records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as w:
        for r in val_records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


class ChatCPODataset(Dataset):
    def __init__(self, original_records, tokenizer, max_tokens_count, sample_rate):
        self.records = []
        for record in tqdm(original_records):

            prompt_messages = record["prompt"]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=False
            )
            prompt = prompt.replace(tokenizer.bos_token, "")
            prompt_tokens = tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=True
            )
            chosen_tokens = tokenizer(record["chosen"][0]["content"])["input_ids"]
            rejected_tokens = tokenizer(record["rejected"][0]["content"])["input_ids"]

            if len(prompt_tokens) + len(chosen_tokens) > max_tokens_count - 10:
                continue
            if len(prompt_tokens) + len(rejected_tokens) > max_tokens_count - 10:
                continue

            self.records.append(
                {
                    "prompt": prompt,
                    "chosen": record["chosen"][0]["content"],
                    "rejected": record["rejected"][0]["content"],
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def train():
    with open(CONFIG_FILE) as r:
        config = json.load(r)

    config["model_name"] = args.base_model_name
    config["cpo"]["cpo_alpha"] = args.cpo_alpha

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    bnb_config = None
    if config["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif config["load_in_8bit"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map=f"cuda:{local_rank}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = config["pad_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.bos_token = config["bos_token"]
    tokenizer.padding_side = "left"

    if os.path.exists(CUSTOM_TEMPLATE_PATH):
        with codecs.open(CUSTOM_TEMPLATE_PATH, "r", "utf-8") as f:
            tokenizer.chat_template = json.load(f)

    if config["load_in_4bit"] or config["load_in_8bit"]:
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", False),
        )
    elif config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = config["lora"]
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
        if model.config.tie_word_embeddings and "lm_head" in config["lora"].get(
            "modules_to_save", []
        ):
            assert (
                "lm_head" not in config["lora"]["modules_to_save"]
                or "embed_tokens" not in config["lora"]["modules_to_save"]
            )
            print("Tie embeddings")
            print(model)
            model.base_model.model.model.embed_tokens.weight = (
                model.base_model.model.lm_head.modules_to_save["default"].weight
            )

    train_dataset = ChatCPODataset(
        read_jsonl(TRAIN_PATH),
        tokenizer,
        config["max_tokens_count"],
        args.sample_fraction,
    )
    train_dataset = HFDataset.from_list(train_dataset)

    eval_dataset = ChatCPODataset(
        read_jsonl(VAL_PATH),
        tokenizer,
        config["max_tokens_count"],
        args.sample_fraction,
    )
    eval_dataset = HFDataset.from_list(eval_dataset)

    training_args = CPOConfig(
        output_dir=args.output_dir, **config["cpo"], **config["trainer"]
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    trainer.model = trainer.model.merge_and_unload()
    trainer.model.train(False)
    merged_output_dir = args.output_dir + "_merged"
    trainer.model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    print(f"Обучение завершено. Модель сохранена в {merged_output_dir}")


if int(os.environ.get("RANK", 0)) == 0:
    compose_pref_dataset(PREF_CONFIG, TRAIN_PATH, VAL_PATH)
    with open("datasets/saiga_preferences/ready.flag", "w") as f:
        f.write("done")
else:
    wait_for_flag("datasets/saiga_preferences/ready.flag")
train()
