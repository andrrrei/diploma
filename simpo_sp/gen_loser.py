import os
import sys
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shard-id", type=int, default=0, help="Индекс текущего шарда")
parser.add_argument("--num-shards", type=int, default=1, help="Общее количество шардов")
args = parser.parse_args()

dataset_name = "IlyaGusev/saiga_preferences"
model_path = "models/qwen2.5-3b-instruct-100-gm-60-saiga-5-0.05-rag_merged"
output_path = f"datasets/saiga_pref_selfplay_1_rank{args.shard_id}.jsonl"
max_new_tokens = 4096
temperature = 0.3
generation_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=0.9,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.0,
    num_return_sequences=1,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
RUADAPT = ROOT / "ruadapt" / "ruadapt"
LLMTF = RUADAPT / "evaluation" / "llmtf_open"
for path in [ROOT, RUADAPT, LLMTF]:
    sys.path.append(str(path))

from llmtf.model import VLLMModel

device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

model = VLLMModel(
    device_map="cuda:0",
    disable_sliding_window=True,
    enable_prefix_caching=True,
    max_seq_len_to_capture=8192,
)
model.from_pretrained(model_path)
model.generation_config = generation_config

ds = load_dataset(dataset_name, split="train")
ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)

new_records = []
for example in tqdm(ds, desc=f"[Rank {args.shard_id}] Generating rejected"):
    prompt_messages = example["prompt"]
    try:
        _, generated, _ = model.generate(prompt_messages, generation_config)
    except Exception as e:
        print(f"[Rank {args.shard_id}] Ошибка генерации: {e}")
        continue

    if not generated or not isinstance(generated, str) or not generated.strip():
        continue

    new_row = dict(example)
    new_row["rejected"] = [{"content": generated.strip(), "role": "assistant"}]
    new_records.append(new_row)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for r in new_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(
    f"Ранг {args.shard_id} завершил генерацию. Сохранено {len(new_records)} примеров в {output_path}"
)
