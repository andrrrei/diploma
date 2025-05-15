import os
import json
import codecs
import random
import argparse

import torch
from datasets import load_from_disk, load_dataset
from datasets import Dataset

from dataset import ChatDataset
from utils import read_jsonl

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser,
)

from peft import get_peft_model, LoraConfig
from peft import prepare_model_for_kbit_training

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument("min_good_score", type=int)
parser.add_argument("ood_fraction", type=float)
parser.add_argument("base_model_name", type=str)
parser.add_argument("output_dir", type=str)
args = parser.parse_args()


def prepare_grounded_rag_dataset(min_good_score: int, ood_fraction: float):
    dataset = load_from_disk("datasets/grounded_rag_ru_v2_scored")
    val_from_hf = load_dataset("Vikhrmodels/Grounded-RAG-RU-v2", split="test")

    train_split = dataset
    good_samples = [
        s for s in train_split if s["type"] == "good" and s["score"] >= min_good_score
    ]
    ood_samples = [s for s in train_split if s["type"] == "ood"]
    random.shuffle(ood_samples)

    ood_limit = int(len(good_samples) * ood_fraction)
    selected_ood = ood_samples[:ood_limit]

    final_train = good_samples + selected_ood
    random.shuffle(final_train)

    print(f"good samples (score ≥ {min_good_score}): {len(good_samples)}")
    print(f"ood samples used ({ood_fraction*100:.0f}%): {len(selected_ood)}")
    print(f"total train: {len(final_train)}, val: {len(val_from_hf)}")

    path = "datasets/Grounded-RAG-RU-v2-filtered"
    os.makedirs(path, exist_ok=True)

    train_path = os.path.join(path, "train.jsonl")
    val_path = os.path.join(path, "val.jsonl")

    if int(os.environ.get("RANK", 0)) == 0:
        Dataset.from_list(final_train).to_json(
            train_path, force_ascii=False, lines=True
        )
        val_from_hf.to_json(val_path, force_ascii=False, lines=True)

    return train_path, val_path


class SFTTrainer:
    def __init__(
        self,
        config_file: str,
        train_file: str,
        val_file: str,
        output_dir: str,
        base_model_name: str,
        custom_chat_template_path: str = None,
        sample_rate: float = 1.0,
        seed: int = 42,
    ):
        self.config_file = config_file
        self.train_file = train_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.custom_chat_template_path = custom_chat_template_path
        self.sample_rate = sample_rate
        self.seed = seed
        self.base_model_name = base_model_name

        self.config = None
        self.training_args = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.data_collator = None
        self.model = None
        self.trainer = None

    def load_config(self):
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.config["model_name"] = self.base_model_name
        print("Конфигурация загружена")

    def init_training_arguments(self):
        trainer_config = self.config.get("trainer", {})
        trainer_config["output_dir"] = self.output_dir
        parser = HfArgumentParser((TrainingArguments,))
        self.training_args = parser.parse_dict(trainer_config)[0]

    def init_tokenizer(self):
        tokenizer_name = self.config.get("tokenizer_name", self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer.bos_token = self.config["bos_token"]
        self.tokenizer.eos_token = self.config["eos_token"]
        self.tokenizer.pad_token = self.config["pad_token"]

        self.tokenizer.padding_side = "left"
        if self.custom_chat_template_path:
            with codecs.open(self.custom_chat_template_path, "r", "utf-8") as f:
                self.tokenizer.chat_template = json.load(f)

        print("Токенизатор инициализирован")

    def prepare_datasets(self, column="messages"):
        train_records = read_jsonl(self.train_file)
        val_records = read_jsonl(self.val_file)

        only_target_loss = self.config.get("only_target_loss", True)
        max_tokens_count = self.config["max_tokens_count"]

        self.train_dataset = ChatDataset(
            train_records,
            self.tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=self.sample_rate,
            only_target_loss=only_target_loss,
            add_global_eos=False,
            add_global_bos=False,
            column=column,
        )
        self.val_dataset = ChatDataset(
            val_records,
            self.tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=self.sample_rate,
            only_target_loss=only_target_loss,
            add_global_eos=False,
            add_global_bos=False,
            column=column,
        )
        print("Данные подготовлены")

    def init_data_collator(self):
        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8
        )

    def init_model(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        load_in_8bit = bool(self.config.get("load_in_8bit", False))
        load_in_4bit = bool(self.config.get("load_in_4bit", False))
        bnb_config = None

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map=f"cuda:{local_rank}",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        if load_in_4bit or load_in_8bit:
            prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.get(
                    "gradient_checkpointing", False
                ),
            )
        elif self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        if self.config.get("lora"):
            lora_config_obj = LoraConfig(**self.config["lora"])
            self.model = get_peft_model(self.model, lora_config_obj)
            if (
                self.model.config.tie_word_embeddings
                and "lm_head" in self.config["lora"]["modules_to_save"]
            ):
                self.model.base_model.model.model.embed_tokens.weight = (
                    self.model.base_model.model.lm_head.modules_to_save[
                        "default"
                    ].weight
                )

        print("Модель инициализирована")

    def setup_trainer(self):
        self.training_args.output_dir = self.output_dir
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
        )
        if len(self.trainer.label_names) == 0:
            self.trainer.label_names.append("labels")
        print("Trainer настроен")

    def run_training(self):
        try:
            self.trainer.train()
        finally:
            if int(os.environ.get("RANK", 0)) == 0:
                try:
                    self.trainer.model = self.trainer.model.merge_and_unload()
                except Exception as e:
                    print("Ошибка при merge_and_unload:", e)

                self.trainer.model.train(False)
                merged_output_dir = self.output_dir + "_merged"
                self.trainer.model.save_pretrained(merged_output_dir)
                self.tokenizer.save_pretrained(merged_output_dir)
                print(f"Обучение завершено. Модель сохранена в {merged_output_dir}")

    def train(self, column="messages"):
        self.load_config()
        self.init_training_arguments()
        self.init_tokenizer()
        self.prepare_datasets(column=column)
        self.init_data_collator()
        self.init_model()
        self.setup_trainer()
        self.run_training()


dataset_path_train, dataset_path_val = prepare_grounded_rag_dataset(
    min_good_score=args.min_good_score, ood_fraction=args.ood_fraction
)

trainer = SFTTrainer(
    config_file="SFT_RAG_1/train_config_3b.json",
    train_file=dataset_path_train,
    val_file=dataset_path_val,
    output_dir=args.output_dir,
    base_model_name=args.base_model_name,
    custom_chat_template_path="configs/qwen_2.5_instruct_no_system.json",
)
trainer.train(column="conversation")
