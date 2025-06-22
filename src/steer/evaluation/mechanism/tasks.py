"""Definition of the Task class and loading of default tasks.
This benchmark tests how good is an LLM at selecting the correct next step."""

import json
import os
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator

import wandb
from steer.logger import setup_logger

logger = setup_logger(__name__)


class Task(BaseModel):
    id: str
    rxn: str
    steps: List[str]
    step_options: List[List[str]]  # Possible moves at each step
    expert_description:str

    def evaluate(self, data: List[List[float]]) -> Tuple[List[float], List[float]]:  # type: ignore
        """data is a list of lists. list[0] is always correct one."""
        gt, lm = [], []
        for i, so in enumerate(self.step_options):
            gt_scores = [10.0] + [0.0] * len(so)
            lm_scores = data[i]
            gt.extend(gt_scores)
            lm.extend(lm_scores)
            logger.info(f"\ngt: {gt_scores}\nlm: {lm_scores}")

            corr = np.corrcoef(gt_scores, lm_scores)[0, 1]
            wandb.log({f"corr_{self.id}_{i}": corr})  # type: ignore
            logger.debug(f"r: {corr:.4f}. Depth: {i}. Good-avg(bad): {lm_scores[0]-np.mean(lm_scores[1:])}")
        return gt, lm

    @classmethod
    def load_from_json(cls, data):
        """Load a task from a json object."""
        return cls(**data)


def load_default_tasks(path=None):
    """Load the default tasks from the benchmark.json file."""
    if path is None:
        path = os.path.dirname(__file__)

    tasks = []
    with open(f"{path}/benchmark.json", "r") as f:
        data = json.load(f)
        for task in data:
            t = Task.load_from_json(task)
            if t is not None:
                tasks.append(t)
    return tasks


if __name__ == "__main__":
    pass
