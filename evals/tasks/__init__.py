from .mmlu import MMLUTask
from .arc import ARCEasyTask, ARCChallengeTask
from .hellaswag import HellaSwagTask
from .gsm8k import GSM8KTask
from .bpb import BPBTask

ALL_TASKS = {
    "mmlu":          MMLUTask,
    "arc_easy":      ARCEasyTask,
    "arc_challenge": ARCChallengeTask,
    "hellaswag":     HellaSwagTask,
    "gsm8k":         GSM8KTask,
    "bpb":           BPBTask,
}
