import os
import pandas as pd
import copy
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset, Dataset

import sys
LLMTF_OPEN_PATH = "ruadapt/ruadapt/evaluation/llmtf_open"
sys.path.append(LLMTF_OPEN_PATH)

from llmtf.base import Task, LLM
from llmtf.metrics import mean
from llmtf.model import VLLMModel
from llmtf.evaluator import Evaluator
from EVAL.MMLU.consts import SUBCATEGORIES, CATEGORIES, SUBCATEGORIES_EN2RU, LANGUAGE_CONFIG

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dataset_single(subject):
    from datasets import DownloadConfig
    config = DownloadConfig()
    config.storage_options = {"timeout": 60}
    return load_dataset(
        MMLU.NLPCORE_HF_PATH,
        name=subject,
        download_mode='reuse_dataset_if_exists',
        download_config=config
    )

def load_dataset_multiprocessing(subjects, num_proc=12):
    if num_proc <= 1:
        datasets = [load_dataset_single(subject) for subject in subjects]
    else:
        with Pool(processes=num_proc) as pool:
            datasets = [ds for ds in pool.map(load_dataset_single, subjects)]
    return datasets

class MMLU(Task):
    NLPCORE_HF_PATH = 'NLPCoreTeam/mmlu_ru'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self._max_new_tokens = 64

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def _per_category_mean(self, results: dict) -> dict:
        subjects = set([res['subject'] for res in results])
        assert len(subjects) == 57
        metric_per_subject = {}
        for subject in subjects:
            metric_per_subject[subject] = mean([res['val'] for res in results if res['subject'] == subject])

        category_to_main_category = {value: key for key, sublist in CATEGORIES.items() for value in sublist}
        subcategories2categories = {key: category_to_main_category[value[0]] for key, value in SUBCATEGORIES.items()}
        subjects = sorted(list(subjects))

        df = pd.DataFrame()
        df['subject'] = subjects
        df['metric'] = [metric_per_subject[s] for s in subjects]
        self.logger.info(df.groupby('subject').mean())
        df['subject'] = df['subject'].apply(lambda x: subcategories2categories[x])
        df = df.groupby('subject').mean()
        self.logger.info(df)

        return float(df.mean())

    def aggregation(self) -> dict:
        return {"acc": self._per_category_mean}

    def evaluate(self, sample, y_pred) -> dict:
        y_true = sample['answer']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        res = y_true == y_pred
        return {'acc': {'val' : res, 'subject': sample['subject']}}

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> tuple[list[dict], list[dict]]:
        messages = []
        samples = []
        subjects = list(SUBCATEGORIES.keys())
        max_samples_per_subject = max_sample_per_dataset // len(subjects) + 1
        subject_datasets = load_dataset_multiprocessing(subjects, 1) #TODO: to params
        for i, dataset in enumerate(tqdm(subject_datasets)):
            subject = subjects[i]

            dataset_test = dataset['test']
            dataset_dev = dataset['dev']

            subject_samples = self._load_dataset(subject, dataset_test, dataset_dev, model, max_len, max_samples_per_subject, few_shot_count)

            subject_messages = [{'messages': s['messages']} for s in subject_samples]
            subject_samples = [{'sample': s['sample']} for s in subject_samples]

            messages += subject_messages
            samples += subject_samples

        for m in messages:
            m['tokens_of_interest'] = self.choices

        return messages, samples


    def _load_dataset(self, subject: str, dataset_test: Dataset, dataset_dev: Dataset, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int):
        assert model.support_method(self.method)
        samples = []
        dataset_test = dataset_test.select(range(min(max_sample_per_dataset, len(dataset_test))))
        for sample in dataset_test:
            samples.append(self._prepare_messages(subject, sample, model, max_len, few_shot_count, dataset_dev))
        return samples

    def _prepare_messages(self, subject: str, sample: dict, model: LLM, max_len: int, few_shot_count: int, few_shot_samples: Dataset) -> list:
        k = min(few_shot_count, len(few_shot_samples))
        int2str = few_shot_samples.features['answer'].int2str

        zero_shot_messages_with_headline = self._create_messages(subject, sample, int2str, add_headline=True, add_answer=False)
        zero_shot_messages_with_headline_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages_with_headline))
        if zero_shot_messages_with_headline_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_with_headline_len} greater then {max_len}. Will be truncated.')

        zero_shot_messages_without_headline = self._create_messages(subject, sample, int2str, add_headline=False, add_answer=False)
        messages = copy.deepcopy(zero_shot_messages_without_headline)
        for i in range(k):
            if i == 0:
                few_shot_messages = self._create_messages(subject, few_shot_samples[i], int2str, add_headline=True, add_answer=True)
                _messages = few_shot_messages + messages
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(_messages))
            else:
                few_shot_messages = self._create_messages(subject, few_shot_samples[i], int2str, add_headline=False, add_answer=True)
                _messages = messages[:-2] + few_shot_messages + messages[-2:]
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(_messages))

            if few_shot_messages_len >= max_len:
                break

            messages = _messages

        sample['answer'] = int2str(sample['answer'])
        sample['subject'] = subject

        return {'messages': messages, 'sample': sample}

    def _create_messages(self, subject, sample, int2str, add_headline=True, add_answer=False):
        q_key = f'question_{self.lang}'
        choice_key = f'choices_{self.lang}'

        headline_prefix = LANGUAGE_CONFIG[self.lang]['headline_prefix']
        headline_postfix = self._get_pretty_subject(subject=subject, lang=self.lang)
        headline = f"{headline_prefix} {headline_postfix}.\n\n"

        answer_prefix = LANGUAGE_CONFIG[self.lang]['answer_prefix'].rstrip()

        q = sample[q_key]
        options = sample[choice_key]
        a = int2str(sample['answer'])

        lettered_options = [f"{x}. {y}" for x, y in zip(["A", "B", "C", "D"], options)]
        q_with_lettered_options = "\n".join([q] + lettered_options)
        if add_headline:
            q_with_lettered_options = headline + q_with_lettered_options

        answer = f'{answer_prefix}'
        if add_answer:
            answer += f' {a}'

        return [{'role': 'user', 'content': q_with_lettered_options}, {'role': 'bot', 'content': answer}]

    def _format_subject(self, subject: str) -> str:
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    def _get_pretty_subject(self, subject: str, lang: str) -> str:
        return self._format_subject({
            "en": subject,
            "ru": SUBCATEGORIES_EN2RU[subject],
        }[self.lang])

    def get_answer(self, sample):
        return ' ' + str(sample['answer'])

class ruMMLU(MMLU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lang = 'ru'

    @classmethod
    def name(cls):
        return 'nlpcoreteam/ruMMLU'


class enMMLU(MMLU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lang = 'en'

    @classmethod
    def name(cls):
        return 'nlpcoreteam/enMMLU'