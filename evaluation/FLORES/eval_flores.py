import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LLMTF = ROOT / "ruadapt" / "ruadapt" / "evaluation" / "llmtf_open"
sys.path.append(str(ROOT))
sys.path.append(str(LLMTF))
sys.path.append(str(ROOT / "ruadapt" / "ruadapt"))


from llmtf.tasks.translation import DaruFlores
from llmtf.evaluator import Evaluator
from llmtf.model import VLLMModel


def run_flores(model_path: str, input_lang: str):
    model = VLLMModel(device_map="cuda:0")
    model.from_pretrained(model_path)
    for input_lang in ["ru", "en"]:
        task = DaruFlores(input_lang)
        ev = Evaluator()
        ev.add_new_task(f"flores_{input_lang}", task)
        result_path = ROOT / "FLORES" / "results" / Path(model_path).name
        os.makedirs(result_path, exist_ok=True)
        ev.evaluate(
            model=model,
            output_dir=result_path,
            datasets_names=["flores"],
            few_shot_count=0,
            batch_size=8,
            max_sample_per_dataset=100000000,
            max_len=32768,
        )
