"""SIFT: marginal-value curation for robot demonstration data."""

from .curate import Curation, curate
from .data import Dataset, Trajectory, make_synthetic, make_validation
from .duplicates import find_duplicates
from .env import EnvConfig, ReachEnv
from .influence import run_influence, tracin
from .judge import ClaudeVLMJudge, CounterfactualJudge

__version__ = "0.2.0"
__all__ = [
    "ClaudeVLMJudge", "CounterfactualJudge", "Curation", "Dataset", "EnvConfig", "ReachEnv", "Trajectory",
    "curate", "find_duplicates", "make_synthetic", "make_validation", "run_influence", "tracin",
]
