import sys
import os
import argparse
import torch
import gc
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUADAPT = ROOT / "ruadapt" / "ruadapt"
LLMTF = RUADAPT / "evaluation" / "llmtf_open"

for path in [ROOT, RUADAPT, LLMTF]:
    sys.path.append(str(path))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = (
    "874c364705747e7ab314ceba89c2029c9a72ab2154664c470eb4ce18c2f0acb0"
)

from EVAL.LIBRA.libra_eval import (
    RuBabilongQA1,
    RuBabilongQA2,
    RuBabilongQA3,
    RuBabilongQA4,
    RuBabilongQA5,
    RuQasper,
)
from llmtf.tasks.nlpcoreteam import enMMLU, ruMMLU
from llmtf.tasks.translation import ruFlores, enFlores
from llmtf.evaluator import Evaluator
from llmtf.model import VLLMModel
from llmtf.tasks.llm_as_a_judge import LLMAsJudge
from llmtf.model import ApiVLLMModel


def run_libra(model: VLLMModel, model_path: str):
    tasks = [
        ("RuBabilongQA1", RuBabilongQA1),
        ("RuBabilongQA2", RuBabilongQA2),
        ("RuBabilongQA3", RuBabilongQA3),
        ("RuBabilongQA4", RuBabilongQA4),
        ("RuBabilongQA5", RuBabilongQA5),
        ("RuQasper", RuQasper),
    ]
    evaluator = Evaluator()
    for subset_name, TaskClass in tasks:
        evaluator.add_new_task(subset_name, TaskClass)
        result_dir = (
            ROOT
            / "EVAL"
            / "LIBRA"
            / "results"
            / f"{subset_name}"
            / Path(model_path).name
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        evaluator.evaluate(
            model=model,
            output_dir=str(result_dir),
            datasets_names=[subset_name],
            few_shot_count=0,
            batch_size=8,
            max_sample_per_dataset=600,
            max_len=32768,
        )


def run_en_mmlu(model: VLLMModel, model_path: str):

    ev = Evaluator()
    ev.add_new_task("enMMLU", enMMLU)

    result_dir = ROOT / "eval" / "MMLU" / "results" / "enMMLU" / Path(model_path).name
    result_dir.mkdir(parents=True, exist_ok=True)

    ev.evaluate(
        model=model,
        output_dir=str(result_dir),
        datasets_names=["enMMLU"],
        few_shot_count=0,
        batch_size=8,
        max_sample_per_dataset=1_000_000_000,
    )


def run_ru_mmlu(model: VLLMModel, model_path: str):

    ev = Evaluator()
    ev.add_new_task("ruMMLU", ruMMLU)

    result_dir = ROOT / "EVAL" / "MMLU" / "results" / "ruMMLU" / Path(model_path).name
    result_dir.mkdir(parents=True, exist_ok=True)

    ev.evaluate(
        model=model,
        output_dir=str(result_dir),
        datasets_names=["ruMMLU"],
        few_shot_count=0,
        batch_size=8,
        max_sample_per_dataset=1_000_000_000,
    )


def run_flores(model: VLLMModel, model_path: str):

    ev = Evaluator()
    ev.add_new_task("flores_ru", ruFlores)
    ev.add_new_task("flores_en", enFlores)

    for task_name in ["flores_ru", "flores_en"]:
        result_dir = (
            ROOT / "EVAL" / "FLORES" / "results" / task_name / Path(model_path).name
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        ev.evaluate(
            model=model,
            output_dir=str(result_dir),
            datasets_names=[task_name],
            few_shot_count=0,
            batch_size=8,
            max_sample_per_dataset=1_000_000_000,
            max_len=32768,
        )


def run_llmaaj(model_path: str):
    out_dir = ROOT / "EVAL" / "LLMAAJ" / "results" / Path(model_path).name
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_file = out_dir / "test_gen.json"

    subprocess.run(
        [
            "python",
            "-m",
            "ruadapt.ruadapt.inference.infer_vllm",
            model_path,
            "ruadapt/ru_llm_arena_hard.json",
            str(gen_file),
            "--infer_for",
            "alpaca_eval",
            "--max_samples",
            "500",
        ],
        check=True,
    )

    refs_root = LLMTF / "llm_as_a_judge_baselines" / "ru_arena-hard-v0.1"
    references = [
        {"model_name": p.stem, "path": str(p)} for p in refs_root.glob("*.json")
    ]

    model = ApiVLLMModel("http://89.169.128.106:6266")
    model.from_pretrained("qwen2.5-72b")

    model.generation_config.max_new_tokens = 1
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0
    model.num_procs = 50

    task = LLMAsJudge(
        model_outputs={
            "model_name": Path(model_path).name,
            "path": str(gen_file),
        },
        references_outputs=references,
    )

    evaluator = Evaluator()
    evaluator.evaluate_dataset(
        task=task,
        model=model,
        output_dir=str(out_dir),
        max_len=4000,
        few_shot_count=0,
        batch_size=256,
        generation_config=None,
        max_sample_per_dataset=1_000_000_000,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["libra", "ru_mmlu", "en_mmlu", "flores", "llmaaj", "llmaaj-only"],
        default=["libra", "ru_mmlu", "en_mmlu", "flores", "llmaaj"],
    )

    args = parser.parse_args()

    model = VLLMModel(
        device_map="cuda:0",
        disable_sliding_window=True,
        enable_prefix_caching=True,
        max_seq_len_to_capture=32768,
    )
    model.from_pretrained(args.model_path)

    if "libra" in args.tasks:
        run_libra(model, args.model_path)

    if "ru_mmlu" in args.tasks:
        run_ru_mmlu(model, args.model_path)

    if "en_mmlu" in args.tasks:
        run_en_mmlu(model, args.model_path)

    if "flores" in args.tasks:
        run_flores(model, args.model_path)

    if "llmaaj" in args.tasks:
        del model
        gc.collect()
        torch.cuda.empty_cache()

        run_llmaaj(args.model_path)
