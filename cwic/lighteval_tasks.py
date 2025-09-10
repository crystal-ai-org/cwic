
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
import lighteval.tasks.default_prompts as prompt


# copied from mmlu_abstract_algebra_leaderboard default task
# changes: name, suite, use "all" subset, loglikelihood_acc_norm_nospace metric
mmlu_leaderboard =LightevalTaskConfig(
    name="mmlu",
    suite=["custom"],
    prompt_function=prompt.mmlu_harness,
    hf_repo="lighteval/mmlu",
    hf_subset="all",
    hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
    evaluation_splits=["test"],
    few_shots_split="dev",
    few_shots_select="sequential",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# copied from winogrande_leaderboard default task
# changes: suite, loglikelihood_acc_norm_nospace metric
winogrande_leaderboard = LightevalTaskConfig(
    name="winogrande",
    suite=["custom"],
    prompt_function=prompt.winogrande,
    hf_repo="winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)


TASKS_TABLE = [
    mmlu_leaderboard,
    winogrande_leaderboard,
]
