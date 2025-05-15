import sys

LLMTF_OPEN_PATH = "ruadapt/ruadapt/evaluation/llmtf_open"
sys.path.append(LLMTF_OPEN_PATH)
RUADAPT_2_PATH = "ruadapt/ruadapt"
sys.path.append(RUADAPT_2_PATH)

import os
import subprocess
import codecs
import json
import codecs
import json

from llmtf.tasks.llm_as_a_judge import LLMAsJudge
from llmtf.evaluator import Evaluator
from llmtf.model import ApiVLLMModel


def run_infer_model(
    model_path, output_dir, alpaca_eval_questions_path, custom_output_path=None
):
    if custom_output_path is None:
        output_path = os.path.join(
            output_dir,
            f"{os.path.basename(output_dir)}_{os.path.basename(model_path)}_ruarena_alpaca_eval.json",
        )
    else:
        output_path = custom_output_path

    print(f"Infer {model_path} to {output_path}")
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"
    call_res = subprocess.call(
        [
            "python",
            "-m",
            "ruadapt.ruadapt.inference.infer_vllm",
            model_path,
            alpaca_eval_questions_path,
            output_path,
            "--infer_for",
            "alpaca_eval",
            "--max_samples",
            str(500),
        ],
        env=my_env,
    )
    if call_res:
        return call_res

    with codecs.open(output_path, "r", "utf-8") as file:
        data = json.load(file)

    for i in range(len(data)):
        data[i]["generator"] = os.path.basename(output_path)[:-5]

    with codecs.open(output_path, "w", "utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return output_path


references = list(
    os.listdir(
        "ruadapt/ruadapt/evaluation/llmtf_open/llm_as_a_judge_baselines/ru_arena-hard-v0.1"
    )
)
references = [
    {
        "model_name": ".".join(ref.split(".")[:-1]),
        "path": os.path.join(
            "ruadapt/ruadapt/evaluation/llmtf_open/llm_as_a_judge_baselines/ru_arena-hard-v0.1",
            ref,
        ),
    }
    for ref in references
]


model_path = "models/qwen2.5-3b-instruct-100-gm-40-opus8-ss_merged"
custom_output_path = os.path.join(
    "LLMAAJ", "results", os.path.basename(model_path), "test_gen.json"
)
os.makedirs(os.path.dirname(custom_output_path), exist_ok=True)
alpaca_eval_questions_path = "ruadapt/ru_llm_arena_hard.json"

data_path = run_infer_model(
    model_path=model_path,
    output_dir=model_path,
    alpaca_eval_questions_path=alpaca_eval_questions_path,
    custom_output_path=custom_output_path,
)

api_base = "http://89.169.128.106:6266"
model = ApiVLLMModel(api_base)
model.from_pretrained("qwen2.5-72b")

model.generation_config.max_new_tokens = 1
model.generation_config.repetition_penalty = 1.0
model.generation_config.do_sample = False
model.generation_config.temperature = 0.0
model.generation_config
model.num_procs = 50

task = LLMAsJudge(
    model_outputs={
        "model_name": os.path.basename(model_path),
        "path": custom_output_path,
    },
    references_outputs=references,
)

evaluator = Evaluator()
evaluator.evaluate_dataset(
    task=task,
    model=model,
    output_dir=model_path,
    max_len=4000,
    few_shot_count=0,
    generation_config=None,
    batch_size=256,
    max_sample_per_dataset=10000000000,
)
