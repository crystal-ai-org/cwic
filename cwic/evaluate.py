
import argparse

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

import models.modelling_cwic


ALL_TASKS = [
    "leaderboard|hellaswag",
    "custom|winogrande",
    "lighteval|piqa",
    "lighteval|openbookqa",
    "lighteval|arc:easy",
    "leaderboard|arc:challenge",
    "custom|mmlu"
]


def main(args):
    """
    Evaluate models using accelerate and transformers as backend.
    """

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="lighteval_tasks.py",
        max_samples=args.max_samples,
        use_chat_template=args.use_chat_template,
    )

    model_config = TransformersModelConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )

    tasks = ""
    for t in args.tasks:
        tasks += f"{t}|0|1,"
    tasks = tasks[:-1]

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    pipeline.save_and_push_results()

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a PBit model on custom tasks using Lighteval")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model checkpoint to load, e.g. "HuggingFaceFW/ablation-model-fineweb-edu"',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save results to. Default: eval_results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size to use for evaluation. Default: None (use model default?)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length for the model. Default: None",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per task. Default: None",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Whether to use chat templates for prompting",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=ALL_TASKS,
        help=f"List of tasks to evaluate on. Default: all. Available: {ALL_TASKS}",
    )

    args = parser.parse_args()
    main(args)
