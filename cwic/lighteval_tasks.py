# ruff: noqa: F405, F403, F401
"""
This evaluation framework is based on Huggingface's FineWeb Ablations:
https://huggingface.co/HuggingFaceFW/ablation-model-fineweb-edu


Run with `max_samples=1000` to speed up large evals.
Most custom prompt changes were in an attempt to improve signal for small models in general.

This file generally creates just a TASKS_TABLE which is then imported by LightEval.

Example usage (lighteval_tasks.py is the path to this file):
===================
lighteval accelerate \
    "model_name=HuggingFaceFW/ablation-model-fineweb-edu" \
    "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu|0|1" \
    --custom_tasks "lighteval_tasks.py" --output_dir [OUTPUTPATH] --max_samples 1000
===================

More info here: https://github.com/huggingface/lighteval?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks
For more info on differences between MMLU implementations: https://huggingface.co/blog/open-llm-leaderboard-mmlu#1001-flavors-of-mmlu
In particular, the default leaderboard MMLU implementation (which uses "A", "B", etc as answer targets) gives generally random results on small/non instruction tuned models.
Instead, we use the full MMLU answer as the target.
"""
import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import arc, openbookqa, piqa_harness, winogrande
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LETTERS = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']


## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = lambda: [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function=hellaswag_prompt,
        hf_repo="hellaswag",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function=winogrande,
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function=piqa_harness,
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=[Metrics.loglikelihood_acc_norm],
        trust_dataset=True,
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function=siqa_prompt,
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function=openbookqa,
        hf_repo="openbookqa",
        hf_subset="main",
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function=arc,
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function=arc,
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc_norm],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function=commonsense_qa_prompt,
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc_norm],
    ),
]


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTERS.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


MMLU_TASK = lambda: LightevalTaskConfig(
    name="mmlu",
    prompt_function=mmlu_prompt,
    hf_repo="lighteval/mmlu",
    hf_subset="all",
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc_norm],
)


def mmlu_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


# Convert to dict for lighteval
TASKS_TABLE = COMMON_SENSE_REASONING_TASKS() + [MMLU_TASK()]
